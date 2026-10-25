#!/usr/bin/env python3
"""gst_finetune_smoke.py — CANONICAL. Real fine-tune smoke for the tiny GST-extraction
LayoutLM (6L/256H/4A, LayoutLMConfig from tiny_gst_config.json).

Trains on the labeled invoice dataset (12 pages, 7 vendors, BIOES labels, boxes 0-1000)
with WHOLE-VENDOR holdout eval. Runs on Colab T4 via:
    colab exec -s <id> -f gst_finetune_smoke_embedded.py --timeout 1200

Outputs to OUT/ = /content/gst_smoke/:
    report.json, best_state.pt, config.json, label2id.json, eval_predictions.jsonl

The embedded variant (generated on the VPS) carries the dataset as base64 so the
single pushed file can materialize it on the VM.
"""
import argparse, base64, json, os, random, time

torch = None  # lazily imported inside main(); module-level so evaluate() can see it
np = None     # lazily imported inside main()

# ==================== data + encoding (stdlib only, importable anywhere) ====================

def build_labels(recs):
    labels = sorted({l for r in recs for l in r["labels"]})
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return labels, label2id, id2label


def encode_record(rec, tok, label2id, max_len=512):
    """Word-level record -> token-level arrays. First subword of each word carries the
    word's label; extra subwords get -100 (ignored in loss). Boxes clamped to valid
    x1<x2, y1<y2 within 0..1000 so LayoutLM's 2D-position indexing never asserts.

    Truncation: HEAD+TAIL window. Pages over budget keep the first chunk (header:
    seller/GSTIN/invoice number/buyer) AND the last chunk (items/tax/totals), dropping
    only the dense middle. This keeps max_len fixed while exposing line items + totals
    to the model (naive tail-only truncation hid them -> 0 F1 on item fields)."""
    # tokenize word-by-word first so we can pick the head+tail window
    word_toks, word_labs, word_boxes = [], [], []
    for w, b, lab_word in zip(rec["words"], rec["boxes"], rec["labels"]):
        toks_w = tok.encode(w, add_special_tokens=False)
        if len(toks_w) == 0:
            toks_w = [tok.unk_token_id]
        x0, y0, x1, y1 = b
        x0 = max(0, min(1000, int(x0))); y0 = max(0, min(1000, int(y0)))
        x1 = max(0, min(1000, int(x1))); y1 = max(0, min(1000, int(y1)))
        if x1 <= x0: x1 = x0 + 1
        if y1 <= y0: y1 = y0 + 1
        word_toks.append(toks_w); word_labs.append(lab_word)
        word_boxes.append([x0, y0, x1, y1])

    budget = max_len - 2  # room for [CLS] and [SEP]
    total = sum(len(t) for t in word_toks)
    windowed = total > budget
    if windowed:
        head_budget = int(budget * 0.55)
        tail_budget = budget - head_budget
        keep = []
        n = 0
        for i, t in enumerate(word_toks):          # head: top of page
            if n + len(t) > head_budget:
                break
            keep.append(i); n += len(t)
        tail = []
        n = 0
        for i in range(len(word_toks) - 1, -1, -1):  # tail: items + totals
            if n + len(word_toks[i]) > tail_budget:
                break
            tail.append(i); n += len(word_toks[i])
        keep += tail[::-1]
    else:
        keep = list(range(len(word_toks)))

    ids, attn, tts, bbox, lab = [], [], [], [], []
    for i in keep:
        toks_w, box, lab_word = word_toks[i], word_boxes[i], word_labs[i]
        for ti, tid in enumerate(toks_w):
            ids.append(tid); attn.append(1); tts.append(0)
            bbox.append(box)
            lab.append(label2id.get(lab_word, label2id["O"]) if ti == 0 else -100)
    n = len(ids)
    truncated = windowed
    ids = [tok.cls_token_id] + ids + [tok.sep_token_id]
    attn = [1] + attn + [1]
    tts = [0] + tts + [0]
    bbox = [[0, 0, 0, 0]] + bbox + [[0, 0, 0, 0]]
    lab = [-100] + lab + [-100]
    pad = max_len - len(ids)
    if pad > 0:
        ids += [tok.pad_token_id] * pad
        attn += [0] * pad
        tts += [0] * pad
        bbox += [[0, 0, 0, 0]] * pad
        lab += [-100] * pad
    return {
        "input_ids": ids[:max_len], "attention_mask": attn[:max_len],
        "token_type_ids": tts[:max_len], "bbox": bbox[:max_len],
        "labels": lab[:max_len], "truncated": truncated, "windowed": windowed,
        "image_id": rec["image_id"], "gold_words": rec["words"], "gold_labels": rec["labels"],
    }


def decode_entities_str(labels_str):
    """BIOES label list (strings, no -100/O mixed) -> [(start, end_inclusive, field)]."""
    ents = []
    i, n = 0, len(labels_str)
    while i < n:
        s = labels_str[i]
        if s == "O":
            i += 1; continue
        pre, _, field = s.partition("-")
        if pre == "S":
            ents.append((i, i, field)); i += 1; continue
        if pre == "B":
            j = i + 1
            while j < n:
                pj, _, fj = labels_str[j].partition("-")
                if fj != field or pj not in ("I", "E"):
                    break
                if pj == "E":
                    j += 1; break
                j += 1
            ents.append((i, j - 1, field)); i = j; continue
        i += 1
    return ents


# ==================== evaluation (stdlib + numpy only) ====================

def eval_metrics(gold_lists, pred_lists):
    """gold_lists/pred_lists: list of label-string sequences (attended, non-ignore).
    Returns token-level P/R/F1 per field (macro over fields), entity-level macro F1,
    and overall token accuracy."""
    per_field_tp = {}; per_field_pred = {}; per_field_gold = {}
    tok_correct = 0; tok_total = 0
    gold_ents_all = []; pred_ents_all = []
    for gold, pred in zip(gold_lists, pred_lists):
        assert len(gold) == len(pred)
        gents = decode_entities_str(gold)
        pents = decode_entities_str(pred)
        gold_ents_all.append(set(gents)); pred_ents_all.append(set(pents))
        for g, p in zip(gold, pred):
            if g == "O" and p == "O":
                continue
            tok_total += 1
            if g == p:
                tok_correct += 1
            gf = g.partition("-")[2]; pf = p.partition("-")[2]
            per_field_gold[gf] = per_field_gold.get(gf, 0) + 1
            per_field_pred[pf] = per_field_pred.get(pf, 0) + 1
            if gf == pf:
                per_field_tp[gf] = per_field_tp.get(gf, 0) + 1
    fields = sorted(set(per_field_gold) | set(per_field_pred))
    per_field_f1 = {}
    for f in fields:
        tp = per_field_tp.get(f, 0); gp = per_field_gold.get(f, 0); pp = per_field_pred.get(f, 0)
        p = tp / pp if pp else 0.0
        r = tp / gp if gp else 0.0
        per_field_f1[f] = 2 * p * r / (p + r) if (p + r) else 0.0
    macro_f1 = sum(per_field_f1.values()) / len(per_field_f1) if per_field_f1 else 0.0
    # entity-level
    etp = sum(len(g & p) for g, p in zip(gold_ents_all, pred_ents_all))
    egold = sum(len(g) for g in gold_ents_all)
    epred = sum(len(p) for p in pred_ents_all)
    ep = etp / epred if epred else 0.0
    er = etp / egold if egold else 0.0
    ent_f1 = 2 * ep * er / (ep + er) if (ep + er) else 0.0
    return {
        "token_acc": tok_correct / tok_total if tok_total else 0.0,
        "macro_field_f1": macro_f1,
        "entity_f1": ent_f1,
        "entity_precision": ep, "entity_recall": er,
        "per_field_f1": per_field_f1,
        "n_gold_entities": egold, "n_pred_entities": epred, "n_entity_matches": etp,
    }


# ==================== main (imports torch/transformers lazily) ====================

# NOTE: no argparse! `colab exec -f` runs the file via the Jupyter kernel launcher,
# which appends its own args to sys.argv and breaks argparse. Use constants / env vars.
DATA_PATH = "/content/gst_smoke_dataset.jsonl"
EPOCHS = int(os.environ.get("GST_EPOCHS", "40"))
BATCH_SIZE = 2
LR = 5e-5
MAX_LEN = 768  # pages measure 454-665 subword tokens; 768 fits ALL with zero truncation
EVAL_EVERY = 10
SEED = 42
OUT_DIR = "/content/gst_smoke"
HOLDOUT_VENDORS = "27AFVFS8057B1ZG,29AAICA3918J1ZE,27AAECB1611P1ZN"


def main():
    global torch, np
    _ensure_dataset(DATA_PATH)

    # --- deps (install guard for a fresh Colab image) ---
    try:
        import torch
        from transformers import LayoutLMConfig, LayoutLMForTokenClassification, BertTokenizerFast
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "transformers", "torch", "numpy"])
        import torch
        from transformers import LayoutLMConfig, LayoutLMForTokenClassification, BertTokenizerFast
    import numpy as np

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "none"
    print(f"[env] torch={torch.__version__} device={device} gpu={gpu_name}", flush=True)

    # --- data ---
    recs = [json.loads(l) for l in open(DATA_PATH)]
    print(f"[data] {len(recs)} records", flush=True)
    holdout = set(HOLDOUT_VENDORS.split(","))
    train_recs = [r for r in recs if r["vendor"] not in holdout]
    eval_recs = [r for r in recs if r["vendor"] in holdout]
    assert train_recs and eval_recs, "empty split"
    print(f"[split] train={len(train_recs)} eval={len(eval_recs)} "
          f"eval_vendors={sorted({r['vendor'] for r in eval_recs})}", flush=True)

    labels, label2id, id2label = build_labels(recs)
    print(f"[labels] {len(labels)} classes", flush=True)

    # --- tokenizer (bert vocab, small download) ---
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    print("[tok] bert-base-uncased loaded", flush=True)

    # --- model ---
    config = LayoutLMConfig(
        vocab_size=30522, hidden_size=256, num_hidden_layers=6,
        num_attention_heads=4, intermediate_size=768,
        max_position_embeddings=MAX_LEN, max_2d_position_embeddings=1024,
        num_labels=len(labels), id2label=id2label, label2id=label2id,
    )
    model = LayoutLMForTokenClassification(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,}", flush=True)
    model.to(device)

    # --- encode all ---
    tr_enc = [encode_record(r, tok, label2id, MAX_LEN) for r in train_recs]
    ev_enc = [encode_record(r, tok, label2id, MAX_LEN) for r in eval_recs]
    n_trunc = sum(1 for e in tr_enc + ev_enc if e["truncated"])
    n_wind = sum(1 for e in tr_enc + ev_enc if e["windowed"])
    if n_trunc:
        print(f"[warn] {n_trunc} records windowed (head+tail) at max_len={MAX_LEN}", flush=True)

    def collate(encs):
        b = {k: torch.tensor([e[k] for e in encs]) for k in
             ("input_ids", "attention_mask", "token_type_ids", "bbox", "labels")}
        return b

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps_per_epoch = (len(tr_enc) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[train] {EPOCHS} epochs x {steps_per_epoch} steps/epoch", flush=True)

    # --- training ---
    train_curve = []
    best = {"f1": -1.0, "epoch": -1}
    t0 = time.time(); n_steps = 0
    for epoch in range(EPOCHS):
        random.shuffle(tr_enc)
        model.train()
        epoch_loss = 0.0
        for i in range(0, len(tr_enc), BATCH_SIZE):
            batch = collate(tr_enc[i:i + BATCH_SIZE])
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            opt.zero_grad(); out.loss.backward(); opt.step()
            epoch_loss += float(out.loss.item()) * batch["input_ids"].size(0)
            n_steps += 1
        epoch_loss /= len(tr_enc)
        train_curve.append(round(epoch_loss, 4))
        msg = f"[ep {epoch+1}/{EPOCHS}] loss {epoch_loss:.4f}"
        if (epoch + 1) % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            m = evaluate(model, ev_enc, collate, id2label, device)
            msg += f" | tok_acc {m['token_acc']:.3f} macroF1 {m['macro_field_f1']:.3f} entF1 {m['entity_f1']:.3f}"
            if m["macro_field_f1"] > best["f1"]:
                best = {"f1": m["macro_field_f1"], "epoch": epoch + 1, "metrics": m}
                os.makedirs(OUT_DIR, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_state.pt"))
                model.config.save_pretrained(OUT_DIR)
                json.dump(label2id, open(os.path.join(OUT_DIR, "label2id.json"), "w"), indent=1)
        print(msg, flush=True)
    dt = time.time() - t0

    # --- final eval on best (reload best weights) ---
    if best["epoch"] > 0:
        model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_state.pt"),
                                         map_location=device))
    final = evaluate(model, ev_enc, collate, id2label, device)

    # --- predictions dump ---
    model.eval()
    preds_out = []
    with torch.no_grad():
        for e in ev_enc:
            batch = collate([e])
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
            pred_ids = logits.argmax(-1)[0]
            attn = batch["attention_mask"][0]
            gold_strs, pred_strs = [], []
            for t_i in range(attn.size(0)):
                if int(attn[t_i]) == 0 or int(e["labels"][t_i]) == -100:
                    continue
                gold_strs.append(id2label[int(e["labels"][t_i])])
                pred_strs.append(id2label[int(pred_ids[t_i])])
            preds_out.append({
                "image_id": e["image_id"],
                "gold": gold_strs, "pred": pred_strs,
            })
    json.dump(preds_out, open(os.path.join(OUT_DIR, "eval_predictions.jsonl"), "w"),
              indent=1)

    report = {
        "device": device, "gpu": gpu_name, "params": n_params,
        "epochs": EPOCHS, "steps": n_steps, "ms_per_step": dt / n_steps * 1000,
        "max_len": MAX_LEN, "truncated_records": n_trunc, "windowed_records": n_wind,
        "train_pages": len(train_recs), "eval_pages": len(eval_recs),
        "eval_vendors": sorted({r["vendor"] for r in eval_recs}),
        "num_labels": len(labels), "label2id": label2id,
        "train_loss_curve": train_curve, "best_epoch": best["epoch"],
        "best_eval": best["metrics"], "final_eval": final,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(report, open(os.path.join(OUT_DIR, "report.json"), "w"), indent=1)
    print(f"[done] saved {OUT_DIR}  ({dt:.1f}s, {dt/n_steps*1000:.0f} ms/step)", flush=True)
    print("FINAL:", json.dumps(final, default=str), flush=True)


def evaluate(model, encs, collate, id2label, device):
    model.eval()
    gold_lists, pred_lists = [], []
    with torch.no_grad():
        for i in range(0, len(encs), 4):
            batch = collate(encs[i:i + 4])
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
            pred_ids = logits.argmax(-1)
            for bi in range(batch["input_ids"].size(0)):
                attn = batch["attention_mask"][bi]
                gold_s, pred_s = [], []
                for t_i in range(attn.size(0)):
                    if int(attn[t_i]) == 0 or int(batch["labels"][bi][t_i]) == -100:
                        continue
                    gold_s.append(id2label[int(batch["labels"][bi][t_i])])
                    pred_s.append(id2label[int(pred_ids[bi][t_i])])
                gold_lists.append(gold_s); pred_lists.append(pred_s)
    return eval_metrics(gold_lists, pred_lists)


# ==================== dataset materialization (embedded variant) ====================

_DATASET_B64 = ""


def _ensure_dataset(path):
    if os.path.exists(path):
        return
    if not _DATASET_B64:
        raise FileNotFoundError(
            f"{path} missing and this build has no embedded dataset. "
            "Use the generated embedded variant.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(base64.b64decode(_DATASET_B64))
    print(f"[data] materialized {path} from embedded base64", flush=True)


if __name__ == "__main__":
    main()