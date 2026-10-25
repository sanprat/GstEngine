#!/usr/bin/env python3
"""build_gst_dataset.py — Build labeled LayoutLM JSONL from the 6 GST invoice PDFs.

Ground truth source: the PDF text layer itself (digitally generated invoices), via
pymupdf page.get_text("words") -> (x0,y0,x1,y1,word). Boxes normalized to 0..1000.
Labels: BIOES over the TINY_GST_MODEL_SPEC field set.
Table cells are labeled by X-COLUMN BANDS derived from header-row token positions.
Header row is located by y-cluster voting (MuPDF line numbers are per-word in tables).
Output: data/gst_smoke_dataset.jsonl + data/gst_smoke_stats.json
"""
import fitz, json, re, glob, os
from collections import defaultdict, Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, 'invoices', 'pdf')
FALLBACK_PDF_DIR = os.path.join(BASE_DIR, '..', 'profiles', 'pybankers', 'cache', 'documents')
IMAGE_DIR = os.path.join(BASE_DIR, 'invoices', 'images')
OCR_OUT_DIR = os.path.join(BASE_DIR, 'invoices', 'ocr_out')
OUT_DIR = os.path.join(BASE_DIR, 'data')

O = 'O'
GSTIN_RE   = re.compile(r'\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]\b')
HSN_RE     = re.compile(r'^[0-9]{6,8}$')
DATE_RE    = re.compile(r'^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$')
MONEY_RE   = re.compile(r'^[-−]?\d[\d,]*\.\d{2}$')          # 117.00 | -30.00 | 1,175.24
INT_RE     = re.compile(r'^[0-9]{1,3}$')
PCT_RE     = re.compile(r'^[0-9]{1,2}(?:\.[0-9]{1,2})?%?$')  # 2.50% | 9.0 | 11.96%

def norm_money(t):
    return t.replace('₹','').replace(',','')

def tokens_of(page):
    ws = page.get_text('words')
    W, H = page.rect.width, page.rect.height
    toks = []
    for w in ws:
        x0, y0, x1, y1, t = w[0], w[1], w[2], w[3], w[4]
        bx = [int(round(x0*1000/W)), int(round(y0*1000/H)), int(round(x1*1000/W)), int(round(y1*1000/H))]
        bx = [max(0, min(1000, v)) for v in bx]
        if bx[2] <= bx[0]: bx[2] = min(1000, bx[0]+1)   # strict x1>x2 guard (LayoutLM)
        if bx[3] <= bx[1]: bx[3] = min(1000, bx[1]+1)
        toks.append(dict(text=t, box=bx, block=w[5], line=w[6],
                         x0=x0, y0=y0, x1=x1, y1=y1))
    return toks

def make_labels(n):
    return [O]*n

def set_span(labels, i, j, f):
    if j < i: return
    L = j - i + 1
    if L == 1:
        labels[i] = f'S-{f}'
    else:
        labels[i] = f'B-{f}'
        for k in range(i+1, j):
            labels[k] = f'I-{f}'
        labels[j] = f'E-{f}'

def line_join(toks):
    g = defaultdict(list)
    for i, t in enumerate(toks):
        g[(t['block'], t['line'])].append(i)
    groups = []
    for key, idxs in g.items():
        idxs.sort(key=lambda i: toks[i]['x0'])
        text = ' '.join(toks[i]['text'] for i in idxs)
        groups.append((idxs, text, toks[idxs[0]]['y0']))
    groups.sort(key=lambda g: g[2])
    return groups

def match_value_tokens(toks, idxs, full_text, key_re):
    m = key_re.search(full_text)
    if not m: return []
    char_pos = len(full_text[:m.end()].rstrip())
    pos, items = 0, []
    for i in idxs:
        txt = toks[i]['text']
        start = full_text.find(txt, pos)
        items.append((i, start, start + len(txt)))
        pos = start + len(txt)
    return [i for i, s, e in items if s >= char_pos]

def find_header_bands(toks, kw_map):
    """Locate table header by y-cluster voting on keyword tokens; return
    bands {(label, occ): x_center} and header y0. kw_map duplicates allowed:
    occurrence order per keyword maps to successive labels."""
    kw_entries = defaultdict(list)
    for kw, lab in kw_map:
        kw_entries[kw].append(lab)
    hits = []
    for i, t in enumerate(toks):
        for kw in kw_entries:
            if t['text'].startswith(kw):
                hits.append((t['y0'], (t['x0']+t['x1'])/2, kw))
                break
    if len(hits) < 3:
        return None, None
    ys = [h[0] for h in hits]
    best_y = max(set(round(y/4)*4 for y in ys),
                 key=lambda y: sum(1 for h in hits if abs(h[0]-y) < 10))
    sel = [h for h in hits if abs(h[0]-best_y) < 10]
    sel.sort(key=lambda h: h[1])
    bands, occ = {}, defaultdict(int)
    for y0, xc, kw in sel:
        lab = kw_entries[kw][min(occ[kw], len(kw_entries[kw])-1)]
        bands[(lab, occ[kw])] = xc
        occ[kw] += 1
    return bands, best_y

def nearest_band(bands, xc):
    return min(bands, key=lambda b: abs(bands[b]-xc))

def page_layout(text):
    lo = text.lower()
    if 'seller name:' in lo or 'evershine cosmic' in lo or 'zepto' in lo:
        return 'zepto'
    if 'original for recipient' in lo and 'sold by' in lo and 'gst registration no' in lo:
        return 'amazon'
    if 'tax invoice' in lo and 'sold by' in lo and 'gstin' in lo:
        return 'flipkart'
    if 'sold by' in lo and 'gstin' in lo:
        return 'flipkart'
    if 'tax invoice' in lo and ('description' in lo or 'sac code' in lo) and ('hsn' in lo or 'sac' in lo) \
       and 'sold by' not in lo:
        return 'synth_gt'
    return None

# ---------------------------------------------------------------- synthetic (GT-labeled)
SYNTH_GT_PATH = os.path.join(BASE_DIR, 'external', 'gst-sample-invoices', 'ground_truth_dataset.json')
_SYNTH_GT_CACHE = None

def _norm_num(t):
    return re.sub(r'[₹,%\s]', '', str(t)).lower()

def _num_eq(token_text, val, tol=0.005):
    try:
        return abs(float(_norm_num(token_text)) - float(val)) < tol
    except (ValueError, TypeError):
        return False

def _find_run(toks, phrase, start=0):
    """Return (i, j) token span of a consecutive run matching phrase words, else None."""
    words = [w for w in re.split(r'\s+', phrase.strip()) if w]
    if not words:
        return None
    n = len(toks)
    for i in range(start, n):
        if _norm_num(toks[i]['text']) != _norm_num(words[0]):
            continue
        k = i
        for w in words[1:]:
            k += 1
            if k >= n or _norm_num(toks[k]['text']) != _norm_num(w):
                break
        else:
            return (i, k)
    return None

def _load_synth_gt():
    global _SYNTH_GT_CACHE
    if _SYNTH_GT_CACHE is None:
        _SYNTH_GT_CACHE = json.load(open(SYNTH_GT_PATH))['invoices']
    return _SYNTH_GT_CACHE

def _resolve_synth_gt(text):
    """Pick the GT record whose invoice number appears in this page's text."""
    for inv in _load_synth_gt():
        invno = inv['invoice_metadata'].get('invoice_number', '')
        if invno and invno.lower() in text.lower():
            return inv
    return None

def label_synth_gt(toks, gt_rec):
    """Label the 4 synthetic GST invoices using the bundled ground-truth JSON.
    Header fields by GT value match; items anchored by HSN then value-scanned
    in reading order (desc = tokens before the HSN token)."""
    n = len(toks)
    labels = make_labels(n)
    used = set()
    if not gt_rec:
        return labels, used
    md = gt_rec.get('invoice_metadata', {})
    sd = gt_rec.get('seller_details', {})
    bd = gt_rec.get('buyer_details', {})
    sm = gt_rec.get('summary', {})

    def one(field, val):
        if val is None:
            return
        want = _norm_num(val)
        for i in range(n):
            if i in used:
                continue
            if want and want in _norm_num(toks[i]['text']):
                set_span(labels, i, i, field); used.add(i); return
        # try reverse containment (token may be a fragment of the value)
        for i in range(n):
            if i in used:
                continue
            t = _norm_num(toks[i]['text'])
            if len(t) >= 3 and t in want:
                set_span(labels, i, i, field); used.add(i); return

    # ---- header (GT value -> field)
    one('SELLER_GSTIN', sd.get('gstin'))
    one('INVOICE_NUMBER', md.get('invoice_number'))
    one('INVOICE_DATE', md.get('invoice_date'))
    one('PLACE_OF_SUPPLY', md.get('place_of_supply'))
    for f, val in (('SELLER_NAME', sd.get('legal_name')),
                   ('BUYER_NAME', bd.get('legal_name'))):
        if val:
            run = _find_run(toks, val)
            if run:
                set_span(labels, run[0], run[1], f); used.update(range(run[0], run[1]+1))

    # ---- items: anchor rows at HSN; desc = contiguous runs of unused non-numeric
    #       tokens before each HSN. Cursor starts AFTER the table-header line so the
    #       document header (TAX INVOICE, addresses, GSTIN...) is never swept in.
    header_line_end = -1
    groups = line_join(toks)
    for idxs, line, _ in groups:
        lo = line.lower()
        if ('hsn' in lo or 'sac' in lo) and any(k in lo for k in ('qty', 'rate', 'taxable', 'description')):
            header_line_end = max(idxs)
            break
    cursor = header_line_end + 1 if header_line_end >= 0 else 0
    NUMERIC_RE = re.compile(r'^[\d,]+(\.\d+)?%?$')
    for item in gt_rec.get('items', []):
        hsn = str(item.get('hsn_code', '') or '').strip()
        hi = None
        for i in range(cursor, n):
            if i in used:
                continue
            t = _norm_num(toks[i]['text'])
            # exact match, or 6-digit token that is a prefix of the 8-digit GT HSN
            if hsn and (t == hsn or (len(t) == 6 and hsn.startswith(t) and not t.isalpha())):
                hi = i; break
        if hi is None:
            continue
        # desc = unused non-numeric tokens on the same visual line as the HSN,
        #       left of the HSN column (y-window: never sweeps the doc header)
        hsn_y, hsn_x = toks[hi]['y0'], toks[hi]['x0']
        desc_idx = [i for i in range(n) if i not in used
                    and abs(toks[i]['y0'] - hsn_y) <= 12
                    and toks[i]['x0'] < hsn_x
                    and not NUMERIC_RE.match(toks[i]['text'].strip())
                    and toks[i]['text'].strip() not in ('#', '|')]
        if desc_idx:
            runs, r = [], None
            for i in sorted(desc_idx):
                if r is None or i != r[1] + 1:
                    if r is not None:
                        runs.append(r)
                    r = [i, i]
                else:
                    r[1] = i
            if r is not None:
                runs.append(r)
            for r in runs:
                set_span(labels, r[0], r[1], 'ITEM_NAME')
                used.update(range(r[0], r[1]+1))
        set_span(labels, hi, hi, 'HSN_SAC'); used.add(hi)
        cursor = hi + 1
        fields = [('QUANTITY', item.get('qty')), ('UNIT_PRICE', item.get('unit_rate')),
                  ('DISCOUNT', item.get('discount')), ('TAXABLE_VALUE', item.get('taxable_value')),
                  ('CGST_RATE', item.get('cgst_rate')), ('CGST_AMOUNT', item.get('cgst_amount')),
                  ('SGST_RATE', item.get('sgst_rate')), ('SGST_AMOUNT', item.get('sgst_amount')),
                  ('IGST_RATE', item.get('igst_rate')), ('IGST_AMOUNT', item.get('igst_amount')),
                  ('TOTAL', item.get('total_amount'))]
        for f, val in fields:
            if val is None or (isinstance(val, (int, float)) and float(val) == 0):
                continue
            for i in range(cursor, min(n, cursor + 40)):
                if i in used:
                    continue
                if _num_eq(toks[i]['text'], val):
                    set_span(labels, i, i, f); used.add(i)
                    cursor = i + 1
                    break

    # ---- grand total
    gt_total = sm.get('grand_total')
    if gt_total is not None:
        for i in range(cursor, n):
            if i in used:
                continue
            if _num_eq(toks[i]['text'], gt_total):
                set_span(labels, i, i, 'TOTAL'); used.add(i); break
    return labels, used

# ---------------------------------------------------------------- zepto
ZEPTO_KW = [('MRP/RSP','MRP'), ('HSN','HSN_SAC'), ('Qty','QUANTITY'),
            ('Rate','UNIT_PRICE'), ('Disc.','DISCOUNT'), ('Taxable','TAXABLE_VALUE'),
            ('CGST','CGST_RATE'), ('S/UT','SGST_RATE'), ('CGST','CGST_AMOUNT'),
            ('S/UT','SGST_AMOUNT'), ('Cess','CESS'), ('Cess','CESS_AMT'),
            ('Total','TOTAL')]

def label_zepto(toks):
    n = len(toks)
    labels = make_labels(n)
    used = set()
    groups = line_join(toks)

    for idxs, line, _ in groups:
        if re.search(r'Seller\s*Name\s*:', line):
            vals = match_value_tokens(toks, idxs, line, re.compile(r'Seller\s*Name\s*:'))
            if vals:
                set_span(labels, vals[0], vals[-1], 'SELLER_NAME'); used.update(vals)
        if re.search(r'GSTIN\s*:', line):
            vals = [i for i in idxs if GSTIN_RE.search(toks[i]['text'])]
            if vals:
                set_span(labels, vals[0], vals[-1], 'SELLER_GSTIN'); used.update(vals)
        if re.search(r'Invoice\s*No\.?\s*:', line):
            vals = match_value_tokens(toks, idxs, line, re.compile(r'Invoice\s*No\.?\s*:'))
            if vals:
                set_span(labels, vals[0], vals[-1], 'INVOICE_NUMBER'); used.update(vals)
        if re.search(r'Date\s*:\s*\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}', line):
            vals = [i for i in idxs if DATE_RE.match(toks[i]['text'])]
            if vals:
                set_span(labels, vals[0], vals[0], 'INVOICE_DATE'); used.add(vals[0])
        if re.search(r'Place\s*Of\s*Supply\s*:', line, re.I):
            vals = [i for i in idxs if re.match(r'^[A-Z]{3,}$', toks[i]['text']) and i not in used]
            if vals:
                set_span(labels, vals[0], vals[0], 'PLACE_OF_SUPPLY'); used.add(vals[0])
        if re.search(r'^Bill\s*To$|^Ship\s*To$', line):
            for i in idxs:
                if toks[i]['text'] in ('TestBuyer', 'Buyer') and i not in used:
                    set_span(labels, i, i, 'BUYER_NAME'); used.add(i)

    bands, hdr_y = find_header_bands(toks, ZEPTO_KW)
    if bands:
        bottom = None
        for idxs, line, y0 in groups:
            if line.startswith('Item Total') or line.startswith('Invoice Value'):
                bottom = y0; break
        if bottom is None: bottom = 1000
        qty_x = bands.get(('QUANTITY',0))
        desc_max_x = min(v for v in bands.values()) - 20
        skip_labs = {'MRP','CESS','CESS_AMT'}
        for i, t in enumerate(toks):
            if not (hdr_y < t['y0'] < bottom) or i in used:
                continue
            txt = t['text']
            xc = (t['x0']+t['x1'])/2
            if HSN_RE.match(txt):
                set_span(labels, i, i, 'HSN_SAC'); used.add(i); continue
            if MONEY_RE.match(norm_money(txt)):
                lab, o = nearest_band(bands, xc)
                if lab not in skip_labs:
                    set_span(labels, i, i, lab); used.add(i)
                continue
            if INT_RE.match(txt):
                if qty_x is not None and abs(xc - qty_x) < 45:
                    set_span(labels, i, i, 'QUANTITY'); used.add(i)
                continue
            if PCT_RE.match(txt):
                lab, o = nearest_band(bands, xc)
                if lab in ('DISCOUNT','CGST_RATE','SGST_RATE','IGST_RATE'):
                    set_span(labels, i, i, lab); used.add(i)
                continue
            if xc < desc_max_x:
                set_span(labels, i, i, 'ITEM_NAME'); used.add(i)
    for idxs, line, _ in groups:
        if line.startswith('Item Total') or line.startswith('Invoice Value'):
            vals = [i for i in idxs if MONEY_RE.match(norm_money(toks[i]['text'])) and i not in used]
            if vals:
                set_span(labels, vals[0], vals[-1], 'TOTAL'); used.update(vals)
    return labels, used

# ---------------------------------------------------------------- flipkart family
FLIP_KW = [('Qty','QUANTITY'), ('Gross','UNIT_PRICE'), ('Discounts','DISCOUNT'),
           ('Taxable','TAXABLE_VALUE'), ('SGST','SGST_AMOUNT'), ('CGST','CGST_AMOUNT'),
           ('IGST','IGST_AMOUNT'), ('Total','TOTAL')]

def label_flipkart(toks):
    n = len(toks)
    labels = make_labels(n)
    used = set()
    groups = line_join(toks)

    for idxs, line, _ in groups:
        if re.search(r'Sold\s*By\s*:', line):
            vals = match_value_tokens(toks, idxs, line, re.compile(r'Sold\s*By\s*:'))
            if not vals:
                # seller name = first following non-label group (skip 'Xxxx :' labels)
                gi = groups.index((idxs, line, _))
                for g2 in groups[gi+1:]:
                    g2_idxs, g2_text, _ = g2
                    if re.search(r'Ship-from Address|GSTIN|PAN\s*:', g2_text):
                        break
                    if g2_text.rstrip().endswith(':') or not g2_text.strip():
                        continue
                    cand = [i for i in g2_idxs if toks[i]['text'] not in (',', '.', '*')]
                    if not cand: continue
                    alpha = [toks[i]['text'] for i in cand if re.search(r'[A-Za-z]', toks[i]['text'])]
                    if len(alpha) >= 2 and not re.search(r'\d', ' '.join(alpha)):
                        vals = cand
                        break
            if vals:
                keep = [i for i in vals if toks[i]['text'] not in (',', '.') and i not in used]
                if keep:
                    set_span(labels, keep[0], keep[-1], 'SELLER_NAME'); used.update(keep)
        if re.search(r'GSTIN\s*[-:]\s*[0-9A-Z]{15}', line):
            vals = [i for i in idxs if GSTIN_RE.search(toks[i]['text'])]
            if vals:
                set_span(labels, vals[0], vals[0], 'SELLER_GSTIN'); used.add(vals[0])
        if re.search(r'Invoice\s*Number\s*#?\s*:?', line) and 'Invoice' in line:
            vals = match_value_tokens(toks, idxs, line, re.compile(r'Invoice\s*Number\s*#?\s*:?'))
            if not vals and 'Original Invoice Number' in line:
                # Instakart variant: value on a later group, skip label groups
                gi = groups.index((idxs, line, _))
                for g2 in groups[gi+1:gi+8]:
                    g2_idxs, g2_text, _ = g2
                    if g2_text.rstrip().endswith(':') or not g2_text.strip():
                        continue
                    cand = [i for i in g2_idxs if
                            re.match(r'^(?=.*\d)[A-Z0-9\-]{6,30}$', toks[i]['text']) and
                            not DATE_RE.match(toks[i]['text'])]
                    if cand:
                        vals = cand
                        break
            if vals:
                set_span(labels, vals[0], vals[-1], 'INVOICE_NUMBER'); used.update(vals)
        if re.search(r'Invoice\s*Date\s*:', line):
            vals = [i for i, t in enumerate(toks) if DATE_RE.match(t['text']) and i not in used]
            if vals:
                set_span(labels, vals[0], vals[0], 'INVOICE_DATE'); used.add(vals[0])
    for i, t in enumerate(toks):
        if t['text'] in ('TestBuyer', 'Buyer') and i not in used:
            set_span(labels, i, i, 'BUYER_NAME'); used.add(i)

    bands, hdr_y = find_header_bands(toks, FLIP_KW)
    if bands:
        bottom = None
        for idxs, line, y0 in groups:
            ls = line.strip()
            if (ls.startswith('Total') or ls.startswith('Grand Total') or ls.startswith('TOTAL')) \
               and hdr_y is not None and y0 > hdr_y + 10:
                bottom = y0; break
        if bottom is None: bottom = 1000
        qty_x = bands.get(('QUANTITY',0))
        desc_max_x = min(v for v in bands.values()) - 20
        for i, t in enumerate(toks):
            if not (hdr_y < t['y0'] < bottom) or i in used:
                continue
            txt = t['text']
            xc = (t['x0']+t['x1'])/2
            if HSN_RE.match(txt):
                set_span(labels, i, i, 'HSN_SAC'); used.add(i); continue
            if txt in ('FSN:', 'HSN/SAC:', 'SGST/UTGST:', 'CGST:', 'IGST:', '1.', '[IMEI/Serial'):
                continue
            if MONEY_RE.match(norm_money(txt)):
                lab, o = nearest_band(bands, xc)
                set_span(labels, i, i, lab); used.add(i); continue
            if INT_RE.match(txt):
                if qty_x is not None and abs(xc - qty_x) < 45:
                    set_span(labels, i, i, 'QUANTITY'); used.add(i)
                continue
            if PCT_RE.match(txt):
                continue
            if xc < desc_max_x:
                set_span(labels, i, i, 'ITEM_NAME'); used.add(i)
    # rate lines (SGST/UTGST: x%  CGST: x%  IGST: x%)
    for idxs, line, _ in groups:
        for key, lab in [('SGST/UTGST','SGST_RATE'), ('CGST','CGST_RATE'), ('IGST','IGST_RATE')]:
            if re.search(key + r'\s*:', line):
                vals = [i for i in idxs if PCT_RE.match(toks[i]['text'])]
                if vals:
                    set_span(labels, vals[0], vals[-1], lab); used.update(vals)
    for idxs, line, _ in groups:
        ls = line.strip()
        if ls.startswith('Grand Total') or ls.startswith('TOTAL:'):
            vals = [i for i in idxs if MONEY_RE.match(norm_money(toks[i]['text'])) and i not in used]
            gi = groups.index((idxs, line, _))
            # money may be on the immediately following line group
            for g2 in groups[gi+1:gi+3]:
                vals += [i for i in g2[0] if MONEY_RE.match(norm_money(toks[i]['text'])) and i not in used]
                if vals: break
            if vals:
                set_span(labels, vals[0], vals[-1], 'TOTAL'); used.update(vals)
    return labels, used

# ---------------------------------------------------------------- amazon
AMZ_KW = [('Unit','UNIT_PRICE'), ('Qty','QUANTITY'), ('Net','TAXABLE_VALUE'),
          ('Amount','TAXABLE_VALUE'), ('Rate','IGST_RATE'), ('Type','IGST_RATE'),
          ('Amount','TAX_AMOUNT'), ('Total','TOTAL'), ('Amount','TOTAL')]

def label_amazon(toks):
    n = len(toks)
    labels = make_labels(n)
    used = set()
    groups = line_join(toks)

    for idxs, line, _ in groups:
        if re.search(r'Sold\s*By\s*:', line):
            vals = match_value_tokens(toks, idxs, line, re.compile(r'Sold\s*By\s*:'))
            if not vals:
                # seller name = first following non-label group (skip 'Xxxx :' labels)
                gi = groups.index((idxs, line, _))
                for g2 in groups[gi+1:]:
                    g2_idxs, g2_text, _ = g2
                    if g2_text.rstrip().endswith(':') or not g2_text.strip():
                        continue  # label or empty line -> keep looking
                    if re.search(r'PAN\s*:|GST\s*Registration', g2_text):
                        break
                    cand = [i for i in g2_idxs if toks[i]['text'] not in (',', '*')]
                    if not cand: continue
                    alpha = [toks[i]['text'] for i in cand if re.search(r'[A-Za-z]', toks[i]['text'])]
                    if len(alpha) >= 2 and not re.search(r'\d', ' '.join(alpha)):
                        vals = cand
                        break
            if vals:
                keep = [i for i in vals if toks[i]['text'] not in (',','*') and i not in used]
                if keep:
                    set_span(labels, keep[0], keep[-1], 'SELLER_NAME'); used.update(keep)
        if re.search(r'GST\s*Registration\s*No\s*:', line):
            vals = [i for i in idxs if GSTIN_RE.search(toks[i]['text'])]
            if vals:
                set_span(labels, vals[0], vals[0], 'SELLER_GSTIN'); used.add(vals[0])
        if re.search(r'Invoice\s*Number\s*:', line):
            vals = match_value_tokens(toks, idxs, line, re.compile(r'Invoice\s*Number\s*:'))
            if vals:
                set_span(labels, vals[0], vals[-1], 'INVOICE_NUMBER'); used.update(vals)
        if re.search(r'Invoice\s*Date\s*:', line):
            vals = [i for i in idxs if DATE_RE.match(toks[i]['text'])]
            if vals:
                set_span(labels, vals[0], vals[0], 'INVOICE_DATE'); used.add(vals[0])
        if re.search(r'Place\s*of\s*supply\s*:', line):
            vals = [i for i in idxs if re.match(r'^[A-Z]{3,}$', toks[i]['text']) and i not in used]
            if vals:
                set_span(labels, vals[0], vals[0], 'PLACE_OF_SUPPLY'); used.add(vals[0])
        if re.search(r'Billing\s*Address\s*:', line):
            vals = match_value_tokens(toks, idxs, line, re.compile(r'Billing\s*Address\s*:'))
            if vals:
                keep = [i for i in vals if toks[i]['text'] not in (',') and i not in used][:2]
                if keep:
                    set_span(labels, keep[0], keep[-1], 'BUYER_NAME'); used.update(keep)
        if re.search(r'TOTAL\s*:', line):
            vals = [i for i in idxs if MONEY_RE.match(norm_money(toks[i]['text'])) and i not in used]
            if vals:
                set_span(labels, vals[0], vals[-1], 'TOTAL'); used.update(vals)
        if re.search(r'Invoice\s*Value\s*:', line):
            vals = [i for i in idxs if MONEY_RE.match(norm_money(toks[i]['text'])) and i not in used]
            if vals:
                set_span(labels, vals[0], vals[-1], 'TOTAL'); used.update(vals)

    bands, hdr_y = find_header_bands(toks, AMZ_KW)
    hsn_idx = [i for i, t in enumerate(toks)
               if HSN_RE.match(t['text']) and i > 0 and toks[i-1]['text'].startswith('HSN')]
    for hidx in hsn_idx:
        set_span(labels, hidx, hidx, 'HSN_SAC'); used.add(hidx)
    if bands:
        bottom = None
        for idxs, line, y0 in groups:
            if line.startswith('TOTAL:') or line.startswith('Amount in Words'):
                bottom = y0; break
        if bottom is None: bottom = 1000
        desc_max_x = min(v for v in bands.values()) - 20
        for i, t in enumerate(toks):
            if not (hdr_y < t['y0'] < bottom) or i in used:
                continue
            txt = t['text']
            xc = (t['x0']+t['x1'])/2
            m = re.match(r'^HSN[:]?(\d{6,8})$', txt)
            if m:
                set_span(labels, i, i, 'HSN_SAC'); used.add(i); continue
            if MONEY_RE.match(norm_money(txt)):
                lab, o = nearest_band(bands, xc)
                if lab in ('UNIT_PRICE','TAXABLE_VALUE','TAX_AMOUNT','TOTAL'):
                    set_span(labels, i, i, lab); used.add(i)
                continue
            if INT_RE.match(txt):
                lab, o = nearest_band(bands, xc)
                if lab == 'QUANTITY':
                    set_span(labels, i, i, 'QUANTITY'); used.add(i)
                continue
            if PCT_RE.match(txt):
                ctx = ' '.join(x['text'] for x in toks[max(0,i-2):min(n,i+3)])
                lab = 'IGST_RATE' if 'IGST' in ctx.upper() else ('CGST_RATE' if 'CGST' in ctx.upper() else ('SGST_RATE' if 'SGST' in ctx.upper() else None))
                if lab:
                    set_span(labels, i, i, lab); used.add(i)
                continue
            if xc < desc_max_x and txt not in ('Sl.','No'):
                set_span(labels, i, i, 'ITEM_NAME'); used.add(i)
    return labels, used

# ---------------------------------------------------------------- driver
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pdfs = sorted(glob.glob(os.path.join(PDF_DIR, '*.pdf')))
    if not pdfs and FALLBACK_PDF_DIR != PDF_DIR:
        pdfs = sorted(glob.glob(os.path.join(FALLBACK_PDF_DIR, '*.pdf')))
        print(f"[dataset] no PDFs in {PDF_DIR} — using fallback {FALLBACK_PDF_DIR} ({len(pdfs)} files)")
    records, skipped = [], []
    seen = set()
    for path in pdfs:
        pid = os.path.splitext(os.path.basename(path))[0]
        doc = fitz.open(path)
        for pno, page in enumerate(doc):
            toks = tokens_of(page)
            text = page.get_text()
            layout = page_layout(text)
            if layout is None:
                skipped.append((pid, pno+1, 'no-layout')); continue
            if layout == 'zepto':
                # Zepto excluded (Oct 23): SGST_AMOUNT column not labeled -> incomplete
                skipped.append((pid, pno+1, 'zepto-excluded')); continue
            if len(toks) < 20:
                skipped.append((pid, pno+1, 'too-few-words')); continue
            if layout == 'amazon': labels, used = label_amazon(toks)
            elif layout == 'synth_gt': labels, used = label_synth_gt(toks, _resolve_synth_gt(text))
            else:                  labels, used = label_flipkart(toks)
            vendor, invno = 'unknown', ''
            for i, lab in enumerate(labels):
                if lab.startswith(('S-SELLER_GSTIN','B-SELLER_GSTIN')):
                    j = i
                    while j+1 < len(labels) and labels[j+1].startswith('I-'):
                        j += 1
                    vendor = get_span_text(toks, i, j)
                if lab == 'S-INVOICE_NUMBER' or lab.startswith('B-INVOICE_NUMBER'):
                    j = i
                    while j+1 < len(labels) and labels[j+1].startswith('I-'):
                        j += 1
                    invno = get_span_text(toks, i, j)
            image_id = f'{pid}_p{pno+1}'
            if image_id in seen: continue
            seen.add(image_id)
            non_o = sum(1 for l in labels if l != O)
            if non_o == 0:
                skipped.append((pid, pno+1, 'no-labels')); continue
            rec = {"image_id": image_id, "vendor": vendor, "doc_type": "invoice",
                   "invoice_number": invno, "words": [t['text'] for t in toks],
                   "boxes": [t['box'] for t in toks], "labels": labels}
            records.append(rec)
        doc.close()
    out = os.path.join(OUT_DIR, 'gst_smoke_dataset.jsonl')
    # Additive merge: build is incremental now. Never lose existing labeled records
    # (original 6 PDFs no longer on disk — JSONL is the durable source of truth).
    old_recs = []
    if os.path.exists(out):
        try:
            old_recs = [json.loads(l) for l in open(out) if l.strip()]
        except Exception:
            old_recs = []
    old_ids = {r['image_id'] for r in old_recs}
    additions = [r for r in records if r['image_id'] not in old_ids]
    if not additions:
        print(f"[dataset] no new records ({len(records)} built, all {len(old_recs)} exist) — keeping {out} unchanged")
        print(json.dumps({"pages": len(old_recs), "built": len(records), "skipped": skipped}, indent=2))
        return
    merged = old_recs + additions
    with open(out, 'w') as f:
        for r in merged:
            f.write(json.dumps(r) + '\n')
    records = merged
    print(f"[dataset] MERGED: +{len(additions)} new (existing {len(old_recs)} kept) = {len(merged)} total")
    labc, vendors = Counter(), Counter()
    for r in records:
        vendors[r['vendor']] += 1
        for l in r['labels']:
            labc[l.split('-')[-1]] += 1
    stats = {"pages": len(records), "vendors": dict(vendors),
             "label_counts": dict(labc.most_common()), "skipped": skipped}
    with open(os.path.join(OUT_DIR, 'gst_smoke_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print('wrote', out)

def get_span_text(toks, a, b):
    return ' '.join(toks[i]['text'] for i in range(a, b+1))

if __name__ == '__main__':
    main()