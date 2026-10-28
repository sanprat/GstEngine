#!/usr/bin/env python3
"""
app_mobile.py — Mobile Testbed Web App for Tiny GST Engine

Allows testing the 13.17 MB on-device model directly from an Android phone:
  - Snap bills with phone camera (<input capture="environment">) or upload
  - RapidOCR + Tiny GST LayoutLM INT8 ONNX inference (~52 ms)
  - Color-coded visual bounding box overlay
  - Mathematical & statutory GST validation (gst_validator.py)
  - 3-tier Section 31 CGST deduplication check (invoice_dedup.py)
  - Direct TallyPrime XML voucher generation & one-click download
"""

from __future__ import annotations
import base64
import json
import os
import re
import socket
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np
import onnxruntime as ort
from transformers import BertTokenizerFast
from rapidocr_onnxruntime import RapidOCR

from extract_to_tally import record_to_invoice, clean_amount
from gst_validator import validate_invoice
from tally_xml import build_import
from invoice_dedup import InvoiceDeduplicator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(BASE_DIR, "smoke_run_v8", "model_int8.onnx")
LABEL2ID_PATH = os.path.join(BASE_DIR, "smoke_run_v8", "label2id.json")
SAMPLES_DIR = os.path.join(BASE_DIR, "external", "gst-sample-invoices")

# Initialize models once at startup
print("[GstEngine] Initializing edge OCR & LayoutLM ONNX INT8 engine...")
label2id = json.load(open(LABEL2ID_PATH))
id2label = {v: k for k, v in label2id.items()}

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", local_files_only=True)

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
onnx_session = ort.InferenceSession(ONNX_PATH, sess_opts, providers=["CPUExecutionProvider"])

ocr_engine = RapidOCR()
dedup_engine = InvoiceDeduplicator()
print("[GstEngine] All engines loaded and ready!")


def get_local_ip() -> str:
    """Detect LAN IP address to share with mobile device."""
    try:
        import subprocess
        out = subprocess.check_output(["ifconfig"], text=True)
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("inet ") and not line.startswith("inet 127."):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def process_invoice_image(img_bgr: np.ndarray, record_to_dedup: bool = False) -> dict:
    t_start = time.perf_counter()
    h, w = img_bgr.shape[:2]

    # 1. Edge OCR
    t0_ocr = time.perf_counter()
    ocr_res, _ = ocr_engine(img_bgr)
    ocr_time_ms = round((time.perf_counter() - t0_ocr) * 1000.0, 1)

    if not ocr_res:
        return {
            "error": "No text detected in image. Please ensure good lighting and clear camera focus."
        }

    words, boxes = [], []
    for bbox, text, conf in ocr_res:
        text = text.strip()
        if not text:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        toks = text.split()
        if len(toks) <= 1:
            bx0 = max(0, min(1000, int(round(min_x * 1000.0 / w))))
            by0 = max(0, min(1000, int(round(min_y * 1000.0 / h))))
            bx1 = max(bx0 + 1, min(1000, int(round(max_x * 1000.0 / w))))
            by1 = max(by0 + 1, min(1000, int(round(max_y * 1000.0 / h))))
            words.append(text)
            boxes.append([bx0, by0, bx1, by1])
        else:
            tot_chars = sum(len(t) for t in toks)
            cur_x = min_x
            box_w = max_x - min_x
            for t in toks:
                ratio = len(t) / max(1, tot_chars)
                word_w = box_w * ratio
                bx0 = max(0, min(1000, int(round(cur_x * 1000.0 / w))))
                by0 = max(0, min(1000, int(round(min_y * 1000.0 / h))))
                bx1 = max(bx0 + 1, min(1000, int(round((cur_x + word_w) * 1000.0 / w))))
                by1 = max(by0 + 1, min(1000, int(round(max_y * 1000.0 / h))))
                words.append(t)
                boxes.append([bx0, by0, bx1, by1])
                cur_x += word_w

    # 2. Subword Tokenization & LayoutLM formatting
    sub_tokens, sub_boxes, word_map = [], [], []
    for w_idx, (word, box) in enumerate(zip(words, boxes)):
        sub_toks = tokenizer.tokenize(word) or ["[UNK]"]
        for st in sub_toks:
            sub_tokens.append(st)
            sub_boxes.append(box)
            word_map.append(w_idx)

    seq_len = min(768, len(sub_tokens) + 2)
    input_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(sub_tokens[:766]) + [tokenizer.sep_token_id]
    bbox = [[0, 0, 0, 0]] + sub_boxes[:766] + [[1000, 1000, 1000, 1000]]
    mask = [1] * len(input_ids)
    type_ids = [0] * len(input_ids)

    pad_len = 768 - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    bbox += [[0, 0, 0, 0]] * pad_len
    mask += [0] * pad_len
    type_ids += [0] * pad_len

    # 3. Model ONNX Forward Pass
    t0_model = time.perf_counter()
    feed = {
        "input_ids": np.array([input_ids], dtype=np.int64),
        "bbox": np.array([bbox], dtype=np.int64),
        "attention_mask": np.array([mask], dtype=np.int64),
        "token_type_ids": np.array([type_ids], dtype=np.int64)
    }
    logits = onnx_session.run(None, feed)[0][0]
    preds = np.argmax(logits[:seq_len], axis=-1)
    model_time_ms = round((time.perf_counter() - t0_model) * 1000.0, 1)

    pred_sub_labels = [id2label[p] for p in preds[1:-1]]

    # Map subword predictions back to word tokens
    word_labels = ["O"] * len(words)
    for st_idx, l in enumerate(pred_sub_labels):
        w_idx = word_map[st_idx]
        if word_labels[w_idx] == "O":
            word_labels[w_idx] = l

    # 4. Extract Structured Invoice Dict & Validate Math
    rec = {"words": words, "boxes": boxes, "labels": word_labels, "image_id": "mobile_scan"}
    invoice = record_to_invoice(rec)
    validation = validate_invoice(invoice)
    invoice["info"] = validation.get("info", {})

    # 5. Statutory Deduplication Check
    dedup_check = dedup_engine.check_duplicate(invoice)
    if record_to_dedup and (not dedup_check["is_duplicate"] or dedup_check["status"] == "NEW"):
        try:
            dedup_engine.record_invoice(invoice)
        except ValueError:
            pass

    # 6. Generate Tally XML Voucher
    tally_xml = build_import(invoice)

    # 7. Prepare Overlay Tokens with Bounding Boxes
    tokens_overlay = []
    for w_text, box, label in zip(words, boxes, word_labels):
        if label != "O":
            tokens_overlay.append({
                "text": w_text,
                "box": box,  # [x0, y0, x1, y1] normalized to 0..1000
                "label": label
            })

    total_time_ms = round((time.perf_counter() - t_start) * 1000.0, 1)

    return {
        "invoice": invoice,
        "validation": validation,
        "dedup": dedup_check,
        "tally_xml": tally_xml,
        "tokens_overlay": tokens_overlay,
        "performance": {
            "ocr_ms": ocr_time_ms,
            "model_ms": model_time_ms,
            "total_ms": total_time_ms,
            "words_count": len(words),
            "labeled_count": len(tokens_overlay)
        }
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Tiny GST Engine — On-Device Scanner</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
  <style>
    :root {
      --primary-color: #2563eb;
      --bg-dark: #0f172a;
    }
    body {
      background: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      padding-bottom: 80px;
    }
    .header-card {
      background: linear-gradient(135deg, #1e3a8a, #3b82f6);
      color: white;
      border-radius: 0 0 24px 24px;
      padding: 24px 20px 20px 20px;
      box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    .camera-btn {
      background: linear-gradient(135deg, #10b981, #059669);
      color: white;
      font-size: 1.15rem;
      font-weight: 600;
      padding: 16px;
      border-radius: 16px;
      box-shadow: 0 10px 20px -3px rgba(16, 185, 129, 0.4);
      border: none;
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      transition: all 0.2s;
    }
    .camera-btn:active {
      transform: scale(0.98);
    }
    .card-custom {
      border: none;
      border-radius: 18px;
      background: white;
      box-shadow: 0 4px 15px -2px rgba(0,0,0,0.06);
      margin-top: 16px;
      overflow: hidden;
    }
    .preview-container {
      position: relative;
      width: 100%;
      background: #000;
      border-radius: 16px;
      overflow: hidden;
      min-height: 200px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    #previewImg {
      width: 100%;
      display: block;
    }
    #bboxCanvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }
    .badge-pill {
      border-radius: 20px;
      padding: 6px 14px;
      font-size: 0.82rem;
      font-weight: 600;
    }
    .table-sm th, .table-sm td {
      font-size: 0.85rem;
      padding: 8px 6px;
    }
    .step-item {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.85rem;
      color: #64748b;
      margin-bottom: 6px;
    }
    .step-item.active {
      color: #2563eb;
      font-weight: 600;
    }
    .step-item.done {
      color: #10b981;
    }
  </style>
</head>
<body>

  <!-- Top Header -->
  <div class="header-card text-center">
    <div class="d-flex justify-content-center align-items-center gap-2 mb-1">
      <i class="bi bi-cpu-fill fs-3"></i>
      <h4 class="mb-0 fw-bold">Tiny GST Engine</h4>
    </div>
    <p class="small text-white-50 mb-2">13.17 MB Edge LayoutLM • Android Native Preview</p>
    <div class="d-flex justify-content-center gap-2">
      <span class="badge bg-white text-dark py-1 px-2 rounded-pill small">~52 ms CPU</span>
      <span class="badge bg-success py-1 px-2 rounded-pill small">100% Offline</span>
      <span class="badge bg-warning text-dark py-1 px-2 rounded-pill small">Tally XML</span>
    </div>
  </div>

  <div class="container px-3 mt-3">

    <!-- Primary Action: Phone Camera -->
    <label class="camera-btn mb-2">
      <i class="bi bi-camera-fill fs-4"></i>
      <span>Snap Bill with Camera</span>
      <input type="file" id="cameraInput" accept="image/*" capture="environment" style="display:none;">
    </label>

    <div class="d-flex gap-2 mb-3">
      <label class="btn btn-outline-secondary w-50 py-2 rounded-3 small fw-semibold">
        <i class="bi bi-upload me-1"></i> Upload Image
        <input type="file" id="fileInput" accept="image/*" style="display:none;">
      </label>
      <button class="btn btn-outline-primary w-50 py-2 rounded-3 small fw-semibold" id="sampleBtn">
        <i class="bi bi-receipt me-1"></i> Try Sample Bill
      </button>
    </div>

    <!-- Live Status / Loading -->
    <div id="loadingCard" class="card card-custom p-3 text-center" style="display:none;">
      <div class="spinner-border text-primary mx-auto mb-2" role="status"></div>
      <h6 class="fw-bold mb-2">Processing Invoice on Edge CPU...</h6>
      <div class="text-start px-3">
        <div id="stepOcr" class="step-item active"><i class="bi bi-circle"></i> 1. Running Edge OCR (text & coordinates)</div>
        <div id="stepModel" class="step-item"><i class="bi bi-circle"></i> 2. Tiny GST 2D Spatial Attention (13 MB ONNX)</div>
        <div id="stepDedup" class="step-item"><i class="bi bi-circle"></i> 3. Statutory Section 31 Deduplication Check</div>
        <div id="stepTally" class="step-item"><i class="bi bi-circle"></i> 4. Generating Idempotent Tally XML Voucher</div>
      </div>
    </div>

    <!-- Invoice Photo & Visual Bounding Box Overlay -->
    <div id="previewCard" class="card card-custom p-2" style="display:none;">
      <div class="preview-container">
        <img id="previewImg" src="" alt="Invoice Preview">
        <canvas id="bboxCanvas"></canvas>
      </div>
      <div class="d-flex justify-content-between align-items-center mt-2 px-1">
        <small class="text-muted"><i class="bi bi-bounding-box"></i> Visual Layout Bounding Boxes</small>
        <span id="perfBadge" class="badge bg-light text-dark border"></span>
      </div>
    </div>

    <!-- Extracted Results Card -->
    <div id="resultCard" class="card card-custom p-3" style="display:none;">
      <div class="d-flex justify-content-between align-items-start mb-3">
        <div>
          <h5 class="fw-bold mb-0" id="partyName">Party Name</h5>
          <small class="text-muted" id="supplierGstin">GSTIN: --</small>
        </div>
        <span id="dedupBadge" class="badge-pill bg-success text-white">NEW</span>
      </div>

      <div class="row g-2 mb-3 bg-light p-2 rounded-3 text-center">
        <div class="col-4">
          <small class="text-muted d-block">Invoice No</small>
          <strong id="invNo">--</strong>
        </div>
        <div class="col-4">
          <small class="text-muted d-block">Date</small>
          <strong id="invDate">--</strong>
        </div>
        <div class="col-4">
          <small class="text-muted d-block">State Code</small>
          <strong id="posCode">--</strong>
        </div>
      </div>

      <!-- Line Items Table -->
      <h6 class="fw-bold small text-muted text-uppercase mb-2">Line Items</h6>
      <div class="table-responsive mb-2">
        <table class="table table-sm table-bordered align-middle mb-0">
          <thead class="table-light">
            <tr>
              <th>Description</th>
              <th>HSN</th>
              <th>Qty</th>
              <th>Rate</th>
              <th>Tax</th>
              <th>Total</th>
            </tr>
          </thead>
          <tbody id="itemsTableBody"></tbody>
        </table>
      </div>

      <!-- Totals Summary -->
      <div class="p-3 rounded-3 mb-3" style="background:#f1f5f9;">
        <div class="d-flex justify-content-between mb-1 small">
          <span>Taxable Amount:</span>
          <strong id="taxableTotal">₹0.00</strong>
        </div>
        <div class="d-flex justify-content-between mb-1 small text-secondary">
          <span>CGST / SGST:</span>
          <span id="taxSplit">₹0.00</span>
        </div>
        <hr class="my-2">
        <div class="d-flex justify-content-between fs-5 fw-bold text-primary">
          <span>Grand Total:</span>
          <span id="grandTotal">₹0.00</span>
        </div>
      </div>

      <!-- Tally Actions -->
      <div class="d-grid gap-2">
        <button id="downloadXmlBtn" class="btn btn-primary fw-semibold py-2">
          <i class="bi bi-download me-1"></i> Download Tally XML
        </button>
        <button id="viewXmlBtn" class="btn btn-outline-secondary py-2 small fw-semibold">
          <i class="bi bi-code-slash me-1"></i> View XML Envelope
        </button>
      </div>

      <!-- XML Modal / Collapse -->
      <div id="xmlBox" class="mt-3 p-2 bg-dark text-light rounded-3 small" style="display:none; max-height:220px; overflow:auto;">
        <pre class="mb-0" style="font-size:0.75rem;"><code id="xmlCode"></code></pre>
      </div>
    </div>

  </div>

  <script>
    let currentData = null;

    const cameraInput = document.getElementById('cameraInput');
    const fileInput = document.getElementById('fileInput');
    const sampleBtn = document.getElementById('sampleBtn');
    const loadingCard = document.getElementById('loadingCard');
    const previewCard = document.getElementById('previewCard');
    const previewImg = document.getElementById('previewImg');
    const bboxCanvas = document.getElementById('bboxCanvas');
    const resultCard = document.getElementById('resultCard');

    cameraInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
    fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
    sampleBtn.addEventListener('click', () => loadSample());

    function handleFile(file) {
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        sendImage(e.target.result);
      };
      reader.readAsDataURL(file);
    }

    function loadSample() {
      fetch('/sample/1')
        .then(r => r.json())
        .then(data => sendImage(data.image_base64))
        .catch(err => alert('Failed to load sample: ' + err));
    }

    function sendImage(base64Data) {
      // Show loading
      loadingCard.style.display = 'block';
      resultCard.style.display = 'none';
      previewCard.style.display = 'block';
      previewImg.src = base64Data;

      fetch('/api/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: base64Data })
      })
      .then(r => r.json())
      .then(data => {
        loadingCard.style.display = 'none';
        if (data.error) {
          alert(data.error);
          return;
        }
        currentData = data;
        renderResults(data);
        drawBoundingBoxes(data.tokens_overlay);
      })
      .catch(err => {
        loadingCard.style.display = 'none';
        alert('Server scan error: ' + err);
      });
    }

    function renderResults(data) {
      resultCard.style.display = 'block';
      const inv = data.invoice;
      const dedup = data.dedup;

      document.getElementById('partyName').textContent = inv.party_name || 'Party Not Specified';
      document.getElementById('supplierGstin').textContent = 'GSTIN: ' + (inv.gstin_supplier || 'None');
      document.getElementById('invNo').textContent = inv.invoice_no || '--';
      document.getElementById('invDate').textContent = inv.date || '--';
      document.getElementById('posCode').textContent = inv.place_of_supply ? ('State ' + inv.place_of_supply) : '--';

      // Deduplication badge
      const dedupBadge = document.getElementById('dedupBadge');
      if (dedup.is_duplicate) {
        dedupBadge.className = 'badge-pill bg-danger text-white';
        dedupBadge.textContent = 'DUPLICATE REJECTED';
      } else {
        dedupBadge.className = 'badge-pill bg-success text-white';
        dedupBadge.textContent = 'NEW INVOICE';
      }

      // Line items
      const tbody = document.getElementById('itemsTableBody');
      tbody.innerHTML = '';
      if (!inv.items || inv.items.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No items extracted</td></tr>';
      } else {
        inv.items.forEach(it => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td><strong>${it.desc || 'Item'}</strong></td>
            <td><small>${it.hsn || '--'}</small></td>
            <td>${it.qty || 1}</td>
            <td>₹${(it.rate || 0).toFixed(2)}</td>
            <td>${it.gst_rate || 0}%</td>
            <td class="fw-bold">₹${(it.taxable || 0).toFixed(2)}</td>
          `;
          tbody.appendChild(tr);
        });
      }

      // Totals
      const taxable = inv.info ? (inv.info.computed_taxable || 0) : 0;
      document.getElementById('taxableTotal').textContent = '₹' + taxable.toFixed(2);
      document.getElementById('grandTotal').textContent = '₹' + (inv.total || 0).toFixed(2);
      document.getElementById('perfBadge').textContent = `${data.performance.total_ms} ms (Model: ${data.performance.model_ms} ms)`;

      // XML View & Download
      document.getElementById('xmlCode').textContent = data.tally_xml;
    }

    function drawBoundingBoxes(tokens) {
      if (!tokens || tokens.length === 0) return;
      const canvas = bboxCanvas;
      const img = previewImg;

      canvas.width = img.clientWidth || img.naturalWidth;
      canvas.height = img.clientHeight || img.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      tokens.forEach(tok => {
        const [bx0, by0, bx1, by1] = tok.box;
        const x = (bx0 / 1000.0) * canvas.width;
        const y = (by0 / 1000.0) * canvas.height;
        const w = ((bx1 - bx0) / 1000.0) * canvas.width;
        const h = ((by1 - by0) / 1000.0) * canvas.height;

        let color = '#3b82f6'; // default blue
        if (tok.label.includes('GSTIN')) color = '#10b981'; // green
        else if (tok.label.includes('TOTAL')) color = '#ef4444'; // red
        else if (tok.label.includes('ITEM')) color = '#8b5cf6'; // purple
        else if (tok.label.includes('TAX')) color = '#f59e0b'; // amber

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
      });
    }

    document.getElementById('downloadXmlBtn').addEventListener('click', () => {
      if (!currentData || !currentData.tally_xml) return;
      const blob = new Blob([currentData.tally_xml], { type: 'text/xml' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const invNo = currentData.invoice.invoice_no || 'invoice';
      a.href = url;
      a.download = `Tally_${invNo}.xml`;
      a.click();
      URL.revokeObjectURL(url);
    });

    document.getElementById('viewXmlBtn').addEventListener('click', () => {
      const box = document.getElementById('xmlBox');
      box.style.display = box.style.display === 'none' ? 'block' : 'none';
    });
  </script>
</body>
</html>
"""


class MobileAppHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif parsed.path.startswith("/sample/"):
            # Provide sample invoice 01
            sample_img_path = os.path.join(SAMPLES_DIR, "GST_Invoice_01_TechHardware_B2B_Intrastate.png")
            if os.path.exists(sample_img_path):
                with open(sample_img_path, "rb") as f:
                    b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"image_base64": b64}).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/scan":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            try:
                data = json.loads(body)
                raw_b64 = data.get("image_base64", "")
                if "," in raw_b64:
                    raw_b64 = raw_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(raw_b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    raise ValueError("Failed to decode image")

                result = process_invoice_image(img, record_to_dedup=data.get("record_dedup", False))
                resp_bytes = json.dumps(result).encode("utf-8")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)
            except Exception as e:
                err_bytes = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_bytes)))
                self.end_headers()
                self.wfile.write(err_bytes)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        # Quiet server logs
        pass


def start_server(port: int = 8000):
    local_ip = get_local_ip()
    server = HTTPServer(("0.0.0.0", port), MobileAppHandler)
    print("\n" + "=" * 65)
    print(f"🚀 Tiny GST Engine — Mobile Testbed Web App is LIVE!")
    print("=" * 65)
    print(f"\n📱 Open this URL on your Android Phone (connected to same Wi-Fi):")
    print(f"\n   👉 \033[1;32mhttp://{local_ip}:{port}\033[0m")
    print(f"   👉 \033[1;34mhttp://localhost:{port}\033[0m (Local Mac browser)\n")
    print("✨ Features on phone:")
    print("   • Tap 'Snap Bill with Camera' to capture physical invoices")
    print("   • Tap 'Try Sample Bill' to test sample invoice immediately")
    print("   • Real-time 13.17 MB ONNX inference (~52 ms)")
    print("   • Instant Download Tally XML to your phone storage")
    print("=" * 65 + "\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[GstEngine] Mobile server stopped.")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    start_server(port)
