"""phone_capture.py — phone-as-scanner over LAN, plus Tally push.

Mounts onto the existing FastAPI app (phone_demo.py):

    GET  /capture        mobile capture page (served by the PC, no app install)
    POST /api/scan       multipart image/PDF  -> extraction JSON (same as /api/extract)
    POST /api/tally      extraction record    -> Tally XML (POST :9000 or import folder)
    GET  /api/tally/ping Tally gateway reachability

Why `capture="environment"` and not getUserMedia():
  getUserMedia requires a secure context (HTTPS or localhost). A phone hitting
  http://192.168.x.x:8000 is NOT a secure context, so live preview is blocked.
  <input type="file" capture="environment"> hands off to the native camera app
  and works fine over plain HTTP — the pragmatic choice for LAN scanning.
"""
import io
import json
import os
import socket

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

# Imported lazily inside handlers to avoid a phone_demo <-> phone_capture cycle.
router = APIRouter()

APP_PIN = os.environ.get("GST_SCAN_PIN", "").strip()  # optional; empty = no PIN

CAPTURE_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0b1020">
<title>GST Scanner — capture</title>
<style>
  :root { --bg:#0b1020; --card:#151b2e; --line:#2a3350; --txt:#e8ecf8; --dim:#93a0c0;
          --ok:#22c55e; --warn:#f59e0b; --bad:#ef4444; --acc:#4f7cff; }
  * { box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
  body { margin:0; background:var(--bg); color:var(--txt); font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; padding:14px 14px 90px; }
  h1 { font-size:17px; margin:0 0 3px; }
  .sub { color:var(--dim); font-size:12.5px; margin-bottom:14px; }
  .btn { display:flex; align-items:center; justify-content:center; gap:8px; width:100%;
         padding:17px; border-radius:14px; border:1px solid var(--line); background:var(--card);
         color:var(--txt); font-size:16px; font-weight:600; margin-bottom:10px; cursor:pointer; }
  .btn.primary { background:var(--acc); border-color:var(--acc); }
  .btn:active { transform:scale(.985); }
  .btn[disabled] { opacity:.45; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:13px; margin-bottom:11px; }
  .row { display:flex; justify-content:space-between; gap:10px; padding:5px 0; font-size:14px; }
  .row .k { color:var(--dim); flex:0 0 auto; }
  .row .v { text-align:right; word-break:break-word; font-variant-numeric:tabular-nums; }
  .v.miss { color:var(--dim); font-style:italic; }
  .v.ok { color:var(--ok); }
  .v.bad { color:var(--bad); }
  .pill { display:inline-block; font-size:11px; padding:2px 8px; border-radius:99px; border:1px solid var(--line); color:var(--dim); }
  .pill.ok { color:var(--ok); border-color:var(--ok); }
  .pill.bad { color:var(--bad); border-color:var(--bad); }
  .thumb { width:64px; height:64px; object-fit:cover; border-radius:9px; border:1px solid var(--line); }
  .qitem { display:flex; gap:11px; align-items:center; padding:9px 0; border-bottom:1px solid var(--line); }
  .qitem:last-child { border-bottom:0; }
  .qname { font-size:13px; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .err { color:var(--bad); font-size:13px; padding:9px 0; }
  #log { font-size:12px; color:var(--dim); font-family:ui-monospace,Menlo,monospace; white-space:pre-wrap; word-break:break-word; }
  .hide { display:none !important; }
</style>
</head><body>
<h1>GST Invoice Scanner</h1>
<div class="sub">Photos are processed on your PC — nothing leaves the local network.</div>

<button class="btn primary" id="shot">📷 Scan invoice</button>
<button class="btn" id="pick">📄 Choose photo / PDF</button>
<input type="file" id="cam" accept="image/*" capture="environment" class="hide">
<input type="file" id="file" accept="image/*,application/pdf" class="hide">

<div id="queueCard" class="card hide">
  <div style="font-size:12px;color:var(--dim);margin-bottom:6px">QUEUE</div>
  <div id="queue"></div>
</div>

<div id="result"></div>
<div id="log"></div>

<script>
const $ = s => document.querySelector(s);
const MAX_EDGE = 2200, QUALITY = 0.85;   // phone photo 12MP -> ~400-900 KB
let pending = [];

$('#shot').onclick = () => $('#cam').click();
$('#pick').onclick = () => $('#file').click();
$('#cam').onchange = e => handleFiles(e.target.files);
$('#file').onchange = e => handleFiles(e.target.files);

function log(m){ $('#log').textContent += m + "\n"; }

async function handleFiles(files){
  for (const f of files) {
    if (!f) continue;
    const item = { name: f.name, blob: f, isPdf: /pdf$/i.test(f.type) || /\.pdf$/i.test(f.name) };
    if (!item.isPdf) {
      try { item.blob = await shrink(f); } catch(e) { log('resize failed, sending original'); }
    }
    pending.push(item);
    renderQueue();
  }
  await drain();
}

// Downscale client-side so uploads stay small on mobile data / slow Wi-Fi.
function shrink(file){
  return new Promise((res, rej) => {
    const img = new Image(), url = URL.createObjectURL(file);
    img.onload = () => {
      const scale = Math.min(1, MAX_EDGE / Math.max(img.width, img.height));
      const c = document.createElement('canvas');
      c.width = Math.round(img.width * scale);
      c.height = Math.round(img.height * scale);
      c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
      URL.revokeObjectURL(url);
      c.toBlob(b => b ? res(b) : rej(new Error('toBlob failed')), 'image/jpeg', QUALITY);
    };
    img.onerror = () => { URL.revokeObjectURL(url); rej(new Error('decode failed')); };
    img.src = url;
  });
}

function renderQueue(){
  $('#queueCard').classList.toggle('hide', !pending.length);
  $('#queue').innerHTML = pending.map(p => `
    <div class="qitem"><div class="qname">${p.name}</div>
    <span class="pill">${p.isPdf ? 'PDF' : Math.round(p.blob.size/1024)+' KB'}</span></div>`).join('');
}

async function drain(){
  while (pending.length) {
    const item = pending[0];
    $('#shot').disabled = $('#pick').disabled = true;
    log(`uploading ${item.name}…`);
    try {
      const fd = new FormData();
      fd.append('file', item.blob, item.name);
      const r = await fetch('/api/scan', { method:'POST', body: fd });
      const j = await r.json();
      if (!r.ok || j.error) throw new Error(j.error || ('HTTP ' + r.status));
      render(j, item.name);
      log(`✓ ${item.name}`);
    } catch (e) {
      log(`✗ ${item.name}: ${e.message}`);
      $('#result').innerHTML = `<div class="card"><div class="err">${item.name}: ${e.message}</div></div>`;
    }
    pending.shift(); renderQueue();
    $('#shot').disabled = $('#pick').disabled = false;
  }
}

const MISS = '<span class="v miss">not detected</span>';
const fmt = (v, cls='') => v === null || v === undefined || v === ''
  ? MISS : `<span class="v ${cls}">${String(v).replace(/</g,'&lt;')}</span>`;

function render(d, name){
  const inv = d.invoice || {};
  const items = inv.items || [];
  const gstin = inv.gstin_supplier || inv.seller_gstin || '';
  const taxTotal = (inv.cgst_total || 0) + (inv.sgst_total || 0) + (inv.igst_total || 0);
  const taxable = inv.taxable ?? items.reduce((s, it) => s + (Number(it.taxable) || 0), 0);
  const warn = (inv.warnings || []).length
    ? `<div style="color:var(--warn);font-size:11.5px;margin-top:7px">⚠ ${inv.warnings.join(' · ')}</div>` : '';
  const gstinPill = gstin.length === 15
    ? `<span class="pill ok">GSTIN ok</span>`
    : `<span class="pill bad">no GSTIN</span>`;
  $('#result').innerHTML = `
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:9px">
      <b style="font-size:14px">${name}</b>${gstinPill}
    </div>
    <div class="row"><span class="k">Invoice no</span>${fmt(inv.invoice_no)}</div>
    <div class="row"><span class="k">Date</span>${fmt(inv.date)}</div>
    <div class="row"><span class="k">Seller</span>${fmt(inv.party_name)}</div>
    <div class="row"><span class="k">Seller GSTIN</span>${fmt(gstin)}</div>
    <div class="row"><span class="k">Buyer</span>${fmt(inv.buyer_name)}</div>
    <div class="row"><span class="k">Place of supply</span>${fmt(inv.place_of_supply)}</div>
    <div class="row"><span class="k">Taxable</span>${fmt(taxable || null)}</div>
    <div class="row"><span class="k">Tax</span>${fmt(taxTotal || null)}</div>
    <div class="row"><span class="k">Total</span>${fmt(inv.total)}</div>
    <div class="row"><span class="k">Items</span><span class="v">${items.length}</span></div>
    ${warn}
  </div>
  <button class="btn primary" id="pushBtn">➡️ Create Tally voucher</button>
  <button class="btn" onclick="$('#result').innerHTML=''">Discard</button>`;
  const btn = document.getElementById('pushBtn');
  if (btn) btn.onclick = () => pushTally(d.xml, !!inv.invoice_no);
}

async function pushTally(xml, haveInvoiceNo){
  if (!xml) { alert('Nothing to push — no voucher XML was produced.'); return; }
  if (!haveInvoiceNo && !confirm('Invoice number was not detected. Push anyway?')) return;
  log('pushing to Tally…');
  try {
    const r = await fetch('/api/tally', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ xml: xml, voucher_type: 'Purchase' })
    });
    const j = await r.json();
    log(r.ok && j.ok ? '✓ Tally: ' + (j.detail || 'voucher created') : '✗ Tally: ' + (j.error || 'failed'));
    if (!r.ok) alert('Tally push failed: ' + (j.error || r.status));
  } catch (e) { log('✗ Tally: ' + e.message); }
}
</script>
</body></html>"""


@router.get("/capture", response_class=HTMLResponse)
def capture_page():
    return HTMLResponse(CAPTURE_HTML)


@router.post("/api/scan")
async def api_scan(file: UploadFile = File(...)):
    """Accept an image or PDF from the phone and return extraction JSON."""
    data = await file.read()
    fn = file.filename or "upload"
    from phone_demo import ocr_image, pdf_words, process
    try:
        if fn.lower().endswith(".pdf") or data[:4] == b"%PDF":
            words, boxes = pdf_words(data)
        else:
            words, boxes = ocr_image(data)
        return JSONResponse(process(words, boxes, source=fn))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/tally")
async def api_tally(payload: dict):
    """Push a Tally voucher.

    Accepts either:
      {"xml": "<ENVELOPE>...", "voucher_type": "Purchase"}   <- from /api/scan
      {"record": {BIOES record}, "voucher_type": "Purchase"} <- server-side build
    """
    from tally_client import post_xml
    host = os.environ.get("TALLY_HOST", "127.0.0.1")
    port = int(os.environ.get("TALLY_PORT", "9000"))
    vtype = payload.get("voucher_type", "Purchase")

    xml = payload.get("xml")
    if not xml:
        record = payload.get("record") or payload
        from extract_to_tally import record_to_import_xml
        try:
            _invoice, xml = record_to_import_xml(record, voucher_type=vtype)
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"xml build failed: {e}"}, status_code=500)
    try:
        resp = post_xml(xml, host=host, port=port)
        return JSONResponse({"ok": True, "detail": f"POSTed to Tally {host}:{port}", "tally_response": str(resp)[:500]})
    except Exception as e:
        out_dir = os.environ.get("TALLY_IMPORT_DIR", "")
        if out_dir and os.path.isdir(out_dir):
            p = os.path.join(out_dir, "gst_voucher.xml")
            with open(p, "w", encoding="utf-8") as f:
                f.write(xml)
            return JSONResponse({"ok": True, "detail": f"Tally unreachable; wrote {p}"})
        return JSONResponse({"ok": False, "error": f"Tally unreachable at {host}:{port}: {e}"}, status_code=502)


@router.get("/api/tally/ping")
def api_tally_ping(host: str = "127.0.0.1", port: int = 9000):
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://{host}:{port}", timeout=4) as r:
            return {"ok": True, "status": r.status}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def print_pairing(port: int):
    """Print the LAN URL + an ASCII QR so the phone never types an IP."""
    url = f"http://{lan_ip()}:{port}/capture"
    print("\n=== PHONE PAIRING ===")
    print(f"Open on your phone: {url}")
    try:
        import qrcode
        q = qrcode.QRCode(border=1)
        q.add_data(url)
        q.make(fit=True)
        q.print_ascii(invert=True)
    except Exception:
        print("(install `qrcode` for a scannable QR code)")
    print("=====================\n")
    return url
