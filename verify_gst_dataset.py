#!/usr/bin/env python3
"""verify_gst_dataset.py — validate the labeled JSONL against known ground truth +
LayoutLM box-safety rules (no CUDA assert landmines)."""
import json, sys, re, os

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'gst_smoke_dataset.jsonl')
if not os.path.exists(PATH):
    PATH = '/opt/hermes/gst-engine/data/gst_smoke_dataset.jsonl'
recs = [json.loads(l) for l in open(PATH)]

print(f"records: {len(recs)}")
errors, warns = [], []

# 1. box validity (LayoutLM golden rule)
for r in recs:
    for i, (w, b) in enumerate(zip(r['words'], r['boxes'])):
        x0, y0, x1, y1 = b
        if not (0 <= x0 <= 1000 and 0 <= y0 <= 1000 and 0 <= x1 <= 1000 and 0 <= y1 <= 1000):
            errors.append(f"{r['image_id']} token {i} '{w}' box out of range {b}")
        if x1 <= x0:
            errors.append(f"{r['image_id']} token {i} '{w}' x1<=x0 {b}")
        if y1 <= y0:
            errors.append(f"{r['image_id']} token {i} '{w}' y1<=y0 {b}")
        if (x1 - x0) >= 1024 or (y1 - y0) >= 1024:
            errors.append(f"{r['image_id']} token {i} '{w}' dim>=1024 {b}")

# 2. helper: extract labeled value per field
def field_values(r, field):
    out, cur = [], []
    for w, l in zip(r['words'], r['labels']):
        if l == 'O':
            if cur: out.append(' '.join(cur)); cur = []
        elif l.startswith('B-') and l.endswith(field):
            cur = [w]
        elif l.startswith(('I-', 'E-')) and l.endswith(field):
            cur.append(w)
        elif l.startswith('S-') and l.endswith(field):
            out.append(w)
    if cur: out.append(' '.join(cur))
    return out

def tot_field(r, field):
    return field_values(r, field)

# 3. ground-truth assertions (Zepto excluded from dataset: SGST_AMOUNT column unlabeled)
fl = [r for r in recs if '682b1f7b36d5' in r['image_id']]
if fl:
    r = fl[0]
    print(f"Flipkart: seller={field_values(r,'SELLER_NAME')} GSTIN={field_values(r,'SELLER_GSTIN')} "
          f"inv={field_values(r,'INVOICE_NUMBER')} total={field_values(r,'TOTAL')}")
    assert field_values(r, 'SELLER_GSTIN') == ['27AAXCS0655F1ZY']
    assert field_values(r, 'INVOICE_NUMBER') == ['LIAAAYT270182425']
    assert any(t in ('136.00',) for t in field_values(r, 'TOTAL')), "grand total 136 missing"
    print("  ✅ Flipkart ground truth verified")

amz = [r for r in recs if 'c32d2798c4d8' in r['image_id']]
if amz:
    for r in amz:
        print(f"Amazon {r['invoice_number']}: seller={field_values(r,'SELLER_NAME')} GSTIN={field_values(r,'SELLER_GSTIN')} "
              f"totals={field_values(r,'TOTAL')}")
    assert any(field_values(r, 'SELLER_GSTIN') == ['27AFVFS8057B1ZG'] for r in amz)
    assert any(field_values(r, 'SELLER_GSTIN') == ['29AAICA3918J1ZE'] for r in amz)
    print("  ✅ Amazon ground truth verified")

# 4. per-vendor coverage report
from collections import Counter
vend = Counter(r['vendor'] for r in recs)
print("\nvendors covered:", dict(vend))
print("distinct vendors:", len(vend))

print(f"\n{'✅ PASS' if not errors else '❌ FAIL'}: {len(errors)} box errors, {len(warns)} warnings")
for e in errors[:10]: print("  ERR", e)
sys.exit(1 if errors else 0)