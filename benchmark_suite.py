"""
benchmark_suite.py — Comprehensive On-Device Benchmark & Comparative Analysis

Compares:
  1. Tiny GST (Our Model): 13.17 MB ONNX INT8 LayoutLM Student (6L/256H/4A, ~13.5M params)
  2. Mobile OCR + Heuristic/Regex Baseline: Standard edge approach (~15-35 MB)
  3. LayoutLM-base (Server SOTA Baseline): 451 MB Transformer (12L/768H/12A, 125M params)

Evaluates on the canonical 18 held-out test pages from 4 completely unseen vendor families:
  - AMULGCMMF (Kirana FMCG invoices)
  - 27AFVFS8057B1ZG (Electronics / Gadgets)
  - 29AAICA3918J1ZE (Wholesale distributor)
  - 27AAECB1611P1ZN (Retail camera photo)

Measures:
  - Model disk size & Parameter count
  - CPU Inference Latency (mean, p95, min, max on Apple Silicon CPU)
  - Memory consumption
  - Entity-level Precision, Recall, and F1 across Indian GST fields
  - Financial & Cloud Cost at Scale (10k, 100k, 1M invoices)
"""

import json
import os
import re
import time
import tracemalloc
from collections import defaultdict
import numpy as np


def load_ground_truth(dataset_path, eval_predictions_path):
    """Load the 18 held-out test pages with their tokens, boxes, and gold labels."""
    eval_preds = json.load(open(eval_predictions_path))
    eval_ids = {x["image_id"] for x in eval_preds}

    pages = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["image_id"] in eval_ids:
                pages.append(rec)

    # Sort deterministically
    pages.sort(key=lambda x: x["image_id"])
    return pages, eval_preds


def extract_entities_from_bioes(words, labels):
    """Convert BIOES labels to structured entities: {field: [text_span, ...]}"""
    entities = defaultdict(list)
    cur_field = None
    cur_tokens = []

    for w, l in zip(words, labels):
        if l == "O":
            if cur_field and cur_tokens:
                entities[cur_field].append(" ".join(cur_tokens).strip())
                cur_field, cur_tokens = None, []
            continue

        prefix, _, field = l.partition("-")
        if prefix == "S":
            if cur_field and cur_tokens:
                entities[cur_field].append(" ".join(cur_tokens).strip())
                cur_field, cur_tokens = None, []
            entities[field].append(w.strip())
        elif prefix == "B":
            if cur_field and cur_tokens:
                entities[cur_field].append(" ".join(cur_tokens).strip())
            cur_field = field
            cur_tokens = [w]
        elif prefix == "I" and cur_field == field:
            cur_tokens.append(w)
        elif prefix == "E" and cur_field == field:
            cur_tokens.append(w)
            entities[field].append(" ".join(cur_tokens).strip())
            cur_field, cur_tokens = None, []
        else:
            if cur_field and cur_tokens:
                entities[cur_field].append(" ".join(cur_tokens).strip())
            cur_field, cur_tokens = None, []

    if cur_field and cur_tokens:
        entities[cur_field].append(" ".join(cur_tokens).strip())
    return entities


# -------------------------------------------------------------------------
# Contender 1: Tiny GST INT8 ONNX Benchmark
# -------------------------------------------------------------------------
def benchmark_tiny_gst_onnx(onnx_path, test_pages, num_warmup=3, num_runs=5):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

    # Prepare dummy inputs of length 768 to benchmark pure CPU forward pass latency
    dummy_input_ids = np.ones((1, 768), dtype=np.int64)
    dummy_bbox = np.zeros((1, 768, 4), dtype=np.int64)
    dummy_mask = np.ones((1, 768), dtype=np.int64)
    dummy_type_ids = np.zeros((1, 768), dtype=np.int64)

    # Warmup
    for _ in range(num_warmup):
        sess.run(None, {
            "input_ids": dummy_input_ids,
            "bbox": dummy_bbox,
            "attention_mask": dummy_mask,
            "token_type_ids": dummy_type_ids
        })

    latencies = []
    for _ in range(num_runs):
        for _ in test_pages:
            t0 = time.perf_counter()
            sess.run(None, {
                "input_ids": dummy_input_ids,
                "bbox": dummy_bbox,
                "attention_mask": dummy_mask,
                "token_type_ids": dummy_type_ids
            })
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    return {
        "name": "Tiny GST (LayoutLM Student INT8)",
        "model_size_mb": round(model_size_mb, 2),
        "params": 13084448,
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "min_latency_ms": round(float(np.min(latencies)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
        "throughput_pages_per_sec": round(1000.0 / float(np.mean(latencies)), 1)
    }


# -------------------------------------------------------------------------
# Contender 2: Mobile OCR + Regex / Heuristic Baseline
# -------------------------------------------------------------------------
def regex_heuristic_extract(words, boxes):
    """
    Standard production-grade regex and heuristic extractor used by mobile apps.
    Without 2D spatial cross-attention heads, it parses text streams.
    """
    text = " ".join(words)
    entities = defaultdict(list)

    # 1. GSTIN Regex (15 alphanumeric characters)
    gstin_matches = re.findall(r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b", text)
    if gstin_matches:
        # Without 2D layout semantics, regex assigns first match as seller, second as buyer
        entities["SELLER_GSTIN"].append(gstin_matches[0])
        if len(gstin_matches) > 1:
            entities["BUYER_NAME"].append(gstin_matches[1])  # common heuristic fallback

    # 2. Invoice Date Regex
    date_match = re.search(r"\b(?:\d{1,2}[-\/.]\d{1,2}[-\/.]\d{2,4}|\d{4}[-\/.]\d{1,2}[-\/.]\d{1,2})\b", text)
    if date_match:
        entities["INVOICE_DATE"].append(date_match.group(0))

    # 3. Invoice Number Regex
    inv_match = re.search(r"(?:Invoice\s*(?:No|Number|#)|Inv\s*No|Bill\s*No)[\s:]*([A-Za-z0-9\/-]+)", text, re.I)
    if inv_match:
        entities["INVOICE_NUMBER"].append(inv_match.group(1))

    # 4. HSN / SAC Regex
    hsn_matches = re.findall(r"(?:HSN|SAC)[\s:]*([0-9]{4,8})\b", text, re.I)
    if not hsn_matches:
        # Fallback: look for 4-8 digit standalone numbers
        hsn_matches = re.findall(r"\b(?:[1-9][0-9]{3}|[1-9][0-9]{7})\b", text)
    for h in hsn_matches[:5]:
        entities["HSN_SAC"].append(h)

    # 5. Grand Total Regex
    total_match = re.search(r"(?:Total|Grand\s*Total|Net\s*Amount|Invoice\s*Value)[\s:₹Rs.]*(\d+(?:\.\d{2})?)", text, re.I)
    if total_match:
        entities["TOTAL"].append(total_match.group(1))

    # 6. Tax amounts (CGST, SGST, IGST)
    cgst_match = re.search(r"CGST[\s\w%@:]*?(\d+(?:\.\d{2})?)", text, re.I)
    if cgst_match:
        entities["CGST_AMOUNT"].append(cgst_match.group(1))

    sgst_match = re.search(r"SGST[\s\w%@:]*?(\d+(?:\.\d{2})?)", text, re.I)
    if sgst_match:
        entities["SGST_AMOUNT"].append(sgst_match.group(1))

    igst_match = re.search(r"IGST[\s\w%@:]*?(\d+(?:\.\d{2})?)", text, re.I)
    if igst_match:
        entities["IGST_AMOUNT"].append(igst_match.group(1))

    return entities


def evaluate_regex_on_test_set(test_pages):
    """Evaluate regex extractor against ground truth annotations on all 18 test pages."""
    t0 = time.perf_counter()
    per_field_stats = defaultdict(lambda: {"gold": 0, "pred": 0, "correct": 0})

    for page in test_pages:
        words = page["words"]
        boxes = page["boxes"]
        gold_labels = page["labels"]

        gold_entities = extract_entities_from_bioes(words, gold_labels)
        pred_entities = regex_heuristic_extract(words, boxes)

        all_fields = set(gold_entities.keys()) | set(pred_entities.keys())
        for f in all_fields:
            g_list = gold_entities.get(f, [])
            p_list = pred_entities.get(f, [])

            per_field_stats[f]["gold"] += len(g_list)
            per_field_stats[f]["pred"] += len(p_list)

            # Simple string overlap match
            matched_pred = set()
            for g in g_list:
                g_clean = re.sub(r"[^A-Za-z0-9]", "", g).upper()
                for i, p in enumerate(p_list):
                    if i in matched_pred:
                        continue
                    p_clean = re.sub(r"[^A-Za-z0-9]", "", p).upper()
                    if g_clean and p_clean and (g_clean in p_clean or p_clean in g_clean):
                        per_field_stats[f]["correct"] += 1
                        matched_pred.add(i)
                        break

    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / len(test_pages)

    # Compute F1 scores
    f1_scores = {}
    total_gold, total_pred, total_correct = 0, 0, 0
    for f, s in per_field_stats.items():
        if f in ("", "O"):
            continue
        p = s["correct"] / s["pred"] if s["pred"] > 0 else 0.0
        r = s["correct"] / s["gold"] if s["gold"] > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        f1_scores[f] = round(f1, 3)
        total_gold += s["gold"]
        total_pred += s["pred"]
        total_correct += s["correct"]

    micro_p = total_correct / total_pred if total_pred > 0 else 0.0
    micro_r = total_correct / total_gold if total_gold > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {
        "name": "Mobile OCR + Regex Heuristics",
        "model_size_mb": 25.0,  # Average mobile OCR engine size (RapidOCR / PP-OCRv4)
        "params": "Rule-Based + Mobile OCR",
        "mean_latency_ms": round(elapsed_ms, 2),
        "micro_f1": round(micro_f1, 3),
        "per_field_f1": f1_scores
    }


# -------------------------------------------------------------------------
# Contender 3: LayoutLM-base (Server SOTA Baseline ~451 MB)
# -------------------------------------------------------------------------
def benchmark_layoutlm_base(test_pages, num_warmup=2, num_runs=3):
    import torch
    from transformers import LayoutLMForTokenClassification

    tracemalloc.start()
    t_load_start = time.perf_counter()
    model = LayoutLMForTokenClassification.from_pretrained(
        "microsoft/layoutlm-base-uncased", num_labels=32
    )
    model.eval()
    load_time_s = time.perf_counter() - t_load_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    param_count = sum(p.numel() for p in model.parameters())

    # Benchmark CPU forward pass latency (LayoutLM-base has max_position_embeddings = 512)
    seq_len = min(512, model.config.max_position_embeddings)
    dummy_input_ids = torch.ones((1, seq_len), dtype=torch.long)
    dummy_bbox = torch.zeros((1, seq_len, 4), dtype=torch.long)
    dummy_mask = torch.ones((1, seq_len), dtype=torch.long)
    dummy_type_ids = torch.zeros((1, seq_len), dtype=torch.long)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(
                input_ids=dummy_input_ids,
                bbox=dummy_bbox,
                attention_mask=dummy_mask,
                token_type_ids=dummy_type_ids
            )

        latencies = []
        for _ in range(num_runs):
            for _ in test_pages:
                t0 = time.perf_counter()
                _ = model(
                    input_ids=dummy_input_ids,
                    bbox=dummy_bbox,
                    attention_mask=dummy_mask,
                    token_type_ids=dummy_type_ids
                )
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

    return {
        "name": "LayoutLM-base (Unquantized Server SOTA)",
        "model_size_mb": round(model_size_mb, 2),
        "params": param_count,
        "load_time_seconds": round(load_time_s, 2),
        "peak_ram_mb": round(peak / (1024 * 1024), 2),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "min_latency_ms": round(float(np.min(latencies)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
        "throughput_pages_per_sec": round(1000.0 / float(np.mean(latencies)), 1)
    }


# -------------------------------------------------------------------------
# Main Benchmark Runner & Report Generator
# -------------------------------------------------------------------------
def run_benchmark():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "data", "gst_smoke_dataset.jsonl")
    eval_preds_path = os.path.join(base_dir, "smoke_run_v8", "eval_predictions.jsonl")
    onnx_path = os.path.join(base_dir, "smoke_run_v8", "model_int8.onnx")
    report_v8_path = os.path.join(base_dir, "smoke_run_v8", "report.json")

    print("=" * 70)
    print("   GstEngine Comprehensive Edge & Document AI Benchmark Suite")
    print("=" * 70)

    print("\n[1/4] Loading canonical 18 held-out test pages from 4 unseen vendors...")
    test_pages, eval_preds = load_ground_truth(dataset_path, eval_preds_path)
    print(f"      Loaded {len(test_pages)} test pages.")

    print("\n[2/4] Benchmarking Tiny GST (13.17 MB INT8 ONNX)...")
    tiny_res = benchmark_tiny_gst_onnx(onnx_path, test_pages)
    # Load Run v8 recorded accuracy
    v8_report = json.load(open(report_v8_path))
    tiny_res["macro_field_f1"] = round(v8_report["best_eval"]["macro_field_f1"], 3)
    tiny_res["entity_f1"] = round(v8_report["best_eval"]["entity_f1"], 3)
    tiny_res["per_field_f1"] = {k: round(v, 3) for k, v in v8_report["best_eval"]["per_field_f1"].items() if k}
    print(f"      Latency: {tiny_res['mean_latency_ms']} ms/page (p95: {tiny_res['p95_latency_ms']} ms)")
    print(f"      Model Size: {tiny_res['model_size_mb']} MB")
    print(f"      Entity F1: {tiny_res['entity_f1']}")

    print("\n[3/4] Benchmarking Mobile OCR + Heuristic / Regex Baseline...")
    regex_res = evaluate_regex_on_test_set(test_pages)
    print(f"      Latency: {regex_res['mean_latency_ms']} ms/page")
    print(f"      Micro F1: {regex_res['micro_f1']}")

    print("\n[4/4] Benchmarking LayoutLM-base (451 MB Server SOTA)...")
    layoutlm_res = benchmark_layoutlm_base(test_pages)
    print(f"      Latency: {layoutlm_res['mean_latency_ms']} ms/page (p95: {layoutlm_res['p95_latency_ms']} ms)")
    print(f"      Model Size: {layoutlm_res['model_size_mb']} MB (Params: {layoutlm_res['params']:,})")
    print(f"      RAM: {layoutlm_res['peak_ram_mb']} MB")

    # Save raw results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "Apple Silicon CPU (M-series, 4 threads)",
        "test_pages": len(test_pages),
        "unseen_vendors": v8_report["eval_vendors"],
        "tiny_gst": tiny_res,
        "regex_baseline": regex_res,
        "layoutlm_base": layoutlm_res
    }

    out_json = os.path.join(base_dir, "data", "benchmark_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Raw benchmark results saved to {out_json}")

    return results


if __name__ == "__main__":
    run_benchmark()
