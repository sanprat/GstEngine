#!/usr/bin/env python3
"""phone_demo.py — GST invoice scanner demo server (mobile-friendly).

Serves a self-contained HTML page on GET / for testing on a phone:
  - upload a photo/PDF of an invoice (camera capture on mobile)
  - OR pick a sample invoice from the benchmark/eval set
Runs the trained LayoutLM model to extract GST fields, then shows the
extracted fields + a Tally XML voucher preview.

Run:  ./venv/bin/python phone_demo.py [--port 8000]
"""
from __future__ import annotations
import argparse, base64, json, os, re, sys, tempfile, time, traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "gst_smoke_dataset.jsonl")
V4_DIR = os.path.join(BASE_DIR, "eval", "smoke_run_v4")
BENCH_DIR = os.path.join(BASE_DIR, "external", "gst-sample-invoices")

# ---------------- model / tokenizer (lazy) ----------------
_model = None
_tokenizer = None
_reader = None

def get_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    import torch
    from transformers import LayoutLMConfig, LayoutLMForTokenClassification, BertTokenizerFast
    cfg = LayoutLMConfig.from_json_file(os.path.join(V4_DIR, "config.json"))
    cfg.num_labels = len(label_list())
    model = LayoutLMForTokenClassification(cfg)
    sd = torch.load(os.path.join(V4_DIR, "best_state.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    _model, _tokenizer = model, tok
    return model, tok

def label_list():
    d = json.load(open(os.path.join(V4_DIR, "label2id.json")))
    return sorted(d, key=d.get)

def get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader

# ---------------- inference ----------------
def word_boxes_to_tokens(words, boxes, tok, max_len=768):
    """words: list[str], boxes: list[[x0,y0,x1,y1] normalized 0-1000]."""
    ids, attn, tts, bbox = [], [], [], []
    for w, b in zip(words, boxes):
        toks = tok.encode(w, add_special_tokens=False)
        if not toks:
            toks = [tok.unk_token_id]
        x0, y0, x1, y1 = (int(round(v)) for v in b)
        x0 = max(0, min(1000, x0)); y0 = max(0, min(1000, y0))
        x1 = max(0, min(1000, x1)); y1 = max(0, min(1000, y1))
        if x1 <= x0: x1 = x0 + 1
        if y1 <= y0: y1 = y0 + 1
        for ti, tid in enumerate(toks):
            if len(ids) >= max_len - 2:
                break
            ids.append(tid); attn.append(1); tts.append(0)
            bbox.append([x0, y0, x1, y1])
        if len(ids) >= max_len - 2:
            break
    ids = [tok.cls_token_id] + ids + [tok.sep_token_id]
    attn = [1] + attn + [1]
    tts = [0] + tts + [0]
    bbox = [[0, 0, 0, 0]] + bbox + [[0, 0, 0, 0]]
    n = len(ids)
    if n < max_len:
        ids += [tok.pad_token_id] * (max_len - n)
        attn += [0] * (max_len - n)
        tts += [0] * (max_len - n)
        bbox += [[0, 0, 0, 0]] * (max_len - n)
    return {
        "input_ids": [ids], "attention_mask": [attn], "token_type_ids": [tts], "bbox": [bbox],
    }

MAX_LEN = 768
WINDOW_OVERLAP_TOKENS = 128


def _predict_window(words, boxes, model, tok, lid):
    batch = word_boxes_to_tokens(words, boxes, tok, max_len=MAX_LEN)
    import torch
    with torch.no_grad():
        logits = model(**{k: torch.tensor(v) for k, v in batch.items()}).logits[0]
    preds = logits.argmax(-1).tolist()
    labels = []
    idx = 1  # skip [CLS]
    avail = len(preds) - 1
    for w in words:
        nt = len(tok.encode(w, add_special_tokens=False) or [tok.unk_token_id])
        if idx >= avail:
            labels.append("O")
            continue
        labels.append(lid.get(preds[idx], "O"))
        idx += nt
    return labels


def predict_labels(words, boxes):
    """Run the model; return a BIOES label per WORD.

    Long documents are processed in overlapping sliding windows (the model was
    trained with head+tail windowing on long pages); per word we prefer any
    non-O prediction over O so fields split across a window edge survive.
    """
    model, tok = get_model()
    lid = {v: k for k, v in json.load(open(os.path.join(V4_DIR, "label2id.json"))).items()}
    counts = [len(tok.encode(w, add_special_tokens=False) or [tok.unk_token_id]) for w in words]
    avail = MAX_LEN - 2

    if sum(counts) <= avail:
        return _predict_window(words, boxes, model, tok, lid)

    # sliding windows over contiguous word ranges
    windows = []
    i = 0
    while i < len(words):
        used, j = 0, i
        while j < len(words) and used + counts[j] <= avail:
            used += counts[j]
            j += 1
        if j == i:  # single word longer than a window: take it alone
            j = i + 1
        windows.append((i, j))
        if j >= len(words):
            break
        ov, k = 0, j
        while k > i and ov < WINDOW_OVERLAP_TOKENS:
            k -= 1
            ov += counts[k]
        i = k if k > i else i + 1

    labels = ["O"] * len(words)
    for a, b in windows:
        wl = _predict_window(words[a:b], boxes[a:b], model, tok, lid)
        for off, l in enumerate(wl):
            pos = a + off
            if l != "O" and labels[pos] == "O":
                labels[pos] = l
    return labels

# ---------------- dataset / samples ----------------
def load_dataset():
    recs = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs

def samples():
    """Return benchmark + dataset samples for the picker."""
    out = []
    for fn in sorted(os.listdir(os.path.join(BENCH_DIR, "internet_samples"))):
        if fn.lower().endswith((".jpg", ".png", ".jpeg")):
            out.append({"name": fn, "kind": "internet"})
    return out[:12]

# ---------------- FastAPI app ----------------
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, io

app = FastAPI(title="GST Invoice Scanner Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GST Invoice Scanner</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:640px;margin:0 auto;padding:16px;background:#0f172a;color:#e2e8f0}
  h1{font-size:22px} .card{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px;margin:10px 0}
  button{background:#2563eb;color:#fff;border:0;border-radius:8px;padding:12px 16px;font-size:15px;margin:4px}
  button:disabled{opacity:.5} .smp{background:#334155;font-size:13px;padding:8px 10px;border-radius:6px;margin:3px;display:inline-block}
  #res{white-space:pre-wrap;font-size:13px} .f{color:#93c5fd} .v{color:#fbbf24}
  #drop{border:2px dashed #475569;border-radius:12px;padding:30px;text-align:center;cursor:pointer}
  #preview{max-width:100%;max-height:280px;margin-top:10px;border-radius:8px}
  .err{color:#fca5a5}
</style></head><body>
<h1>🧾 GST Invoice Scanner</h1>
<div class="card" id="drop">📷 Tap to take a photo or choose a file<br><small>(JPG/PNG/PDF of an invoice)</small>
<input type="file" id="file" accept="image/*,.pdf" style="display:none"></div>
<img id="preview" style="display:none">
<div class="card"><b>Or try a sample:</b><br><span id="samples"></span></div>
<div class="card"><b>Extracted fields</b><div id="res"><i>Waiting for an invoice…</i></div></div>
<script>
const $=id=>document.getElementById(id);
const drop=$('drop'), file=$('file'), res=$('res'), preview=$('preview');
drop.onclick=()=>file.click();
file.onchange=()=>{ if(file.files[0]){ const f=file.files[0];
  if(f.type.startsWith('image/')){preview.src=URL.createObjectURL(f);preview.style.display='block';}
  res.innerHTML='<i>Scanning…</i>'; scan(f); } };
async function scan(file){
  const fd=new FormData(); fd.append('file', file);
  try{
    const r=await fetch('/api/extract',{method:'POST',body:fd});
    const d=await r.json();
    if(d.error) res.innerHTML='<span class="err">'+d.error+'</span>';
    else{ res.innerHTML='';
      (d.fields||[]).forEach(x=>{res.innerHTML+=`<div><span class="f">${x[0]}</span>: <span class="v">${x[1]}</span></div>`;});
      if(d.xml) res.innerHTML+='<hr><details><summary>Show Tally XML</summary><pre>'+d.xml+'</pre></details>';
    }
  }catch(e){ res.innerHTML='<span class="err">Error: '+e+'</span>'; }
}
fetch('/api/samples').then(r=>r.json()).then(s=>{
  const box=$('samples');
  s.forEach(sm=>{ const b=document.createElement('button'); b.className='smp';
    b.textContent=sm.name; b.onclick=()=>{res.innerHTML='<i>Scanning…</i>'; scanSample(sm.name);};
    box.appendChild(b); });
});
async function scanSample(name){
  try{ const r=await fetch('/api/sample/'+encodeURIComponent(name)); const d=await r.json();
    if(d.error) res.innerHTML='<span class="err">'+d.error+'</span>';
    else{ res.innerHTML='';
      (d.fields||[]).forEach(x=>{res.innerHTML+=`<div><span class="f">${x[0]}</span>: <span class="v">${x[1]}</span></div>`;});
      if(d.xml) res.innerHTML+='<hr><details><summary>Show Tally XML</summary><pre>'+d.xml+'</pre></details>';
    }
  }catch(e){ res.innerHTML='<span class="err">Error: '+e+'</span>'; }
}
</script></body></html>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

@app.get("/api/samples")
def api_samples():
    return [s["name"] for s in samples()]

@app.post("/api/extract")
async def api_extract(file: UploadFile = File(...)):
    data = await file.read()
    fn = file.filename or "upload"
    try:
        if fn.lower().endswith(".pdf") or data[:4] == b"%PDF":
            words, boxes = pdf_words(data)
        else:
            words, boxes = ocr_image(data)
        return JSONResponse(process(words, boxes, source=fn))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/sample/{name}")
def api_sample(name: str):
    try:
        path = os.path.join(BENCH_DIR, "internet_samples", name)
        with open(path, "rb") as f:
            data = f.read()
        words, boxes = ocr_image(data)
        return JSONResponse(process(words, boxes, source=name))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ---------------- OCR / pdf ----------------
def pdf_words(data):
    """Extract words + normalized boxes from a PDF byte blob (ALL pages)."""
    import fitz
    doc = fitz.open(stream=data, filetype="pdf")
    words, boxes = [], []
    for page in doc:
        pw, ph = page.rect.width, page.rect.height
        for w in page.get_text("words"):
            x0, y0, x1, y1, txt = w[0], w[1], w[2], w[3], w[4]
            words.append(txt)
            boxes.append([x0 / pw * 1000, y0 / ph * 1000, x1 / pw * 1000, y1 / ph * 1000])
    return words, boxes

def ocr_image(data):
    """OCR an image into words + normalized boxes (0-1000)."""
    import numpy as np
    from PIL import Image
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    reader = get_reader()
    result = reader.readtext(np.array(img))
    words, boxes = [], []
    for bbox, text, conf in result:
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        words.append(text)
        boxes.append([min(xs) / w * 1000, min(ys) / h * 1000, max(xs) / w * 1000, max(ys) / h * 1000])
    return words, boxes

# ---------------- span → invoice → Tally ----------------
def process(words, boxes, source="upload"):
    """words/boxes → predicted BIOES → validated invoice dict → Tally XML."""
    labels = predict_labels(words, boxes)
    record = {"image_id": source, "vendor": "pred", "doc_type": "upload",
              "invoice_number": "", "words": words, "boxes": boxes, "labels": labels}
    # field spans in reading order for display
    from extract_to_tally import field_spans, span_text
    spans = field_spans(record)
    fields = []
    seen = set()
    for a, b, f in spans:
        txt = span_text(record, a, b)
        if not txt.strip():
            continue
        key = (f, a)
        if key in seen:
            continue
        seen.add(key)
        fields.append((f, txt))
    out = {"fields": fields, "source": source}
    try:
        from extract_to_tally import record_to_invoice, record_to_import_xml
        invoice = record_to_invoice(record)
        out["invoice"] = invoice
        _inv2, xml = record_to_import_xml(record, "Purchase")
        out["xml"] = xml
    except Exception as e:
        out["xml_error"] = str(e)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"Serving GST scanner demo on 0.0.0:{args.port} (tunnel: cloudflared tunnel --url http://localhost:{args.port})")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()