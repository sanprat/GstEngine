#!/usr/bin/env python3
"""extract_to_tally.py — Bridge: LayoutLM model output (BIOES-labeled JSONL record)
-> validated invoice dict -> TallyPrime import XML (file-based, ready to drop into
Tally's import folder, or POST to Tally's :9000 HTTP endpoint).

Pipeline:  record -> invoice dict -> gst_validator.validate_invoice()
        -> tally_xml.build_import() -> .xml file / HTTP POST

Invoice dict schema (matches tally_xml.py + gst_validator.py):
  gstin_supplier, invoice_no, date (dd-mm-yyyy), place_of_supply (2-digit state code),
  party_name, items[{hsn, desc, qty, unit, rate, taxable, gst_rate}], total
"""
from __future__ import annotations
import json, re, sys, os
from collections import defaultdict
from datetime import datetime

from gst_validator import validate_invoice, STATE_CODES
from tally_xml import build_import, envelope
from invoice_dedup import InvoiceDeduplicator

# normalise state names ("MAHARASTRA" typo on the real Zepto invoice -> Maharashtra)
STATE_NAME_FIX = {
    "MAHARASTRA": "MAHARASHTRA", "MAHARASHTRA": "MAHARASHTRA",
    "KARNATAKA": "KARNATAKA", "DELHI": "DELHI", "MUMBAI": "MAHARASHTRA",
    "TAMIL NADU": "TAMIL NADU", "TELANGANA": "TELANGANA", "KERALA": "KERALA",
    "GOA": "GOA", "UTTAR PRADESH": "UTTAR PRADESH", "RAJASTHAN": "RAJASTHAN",
}
NAME_TO_CODE = {v: k for k, v in STATE_CODES.items()}

def field_spans(record):
    """Yield (field, start_i, end_i) spans from BIOES labels, in reading order."""
    spans = []
    cur = None
    for i, l in enumerate(record["labels"]):
        if l == "O":
            cur = None
            continue
        pre, _, field = l.partition("-")
        if pre == "B":
            cur = [i, i, field]
        elif pre in ("I", "E") and cur and cur[2] == field:
            cur[1] = i
        elif pre == "S":
            cur = [i, i, field]
            spans.append(tuple(cur)); cur = None
        if cur and (pre == "B" or (pre in ("I", "E") and cur)):
            if l.startswith("S-"):
                spans.append(tuple(cur)); cur = None
            elif pre == "B":
                pass  # keep accumulating
            elif pre == "E":
                spans.append(tuple(cur)); cur = None
    return spans

def span_text(record, a, b):
    return " ".join(record["words"][i] for i in range(a, b + 1)).strip()

def clean_amount(s: str) -> float:
    m = re.search(r"[-+]?\d+(?:,\d+)*(?:\.\d+)?", str(s))
    if m:
        try:
            return float(m.group(0).replace(",", ""))
        except ValueError:
            pass
    return 0.0

HEADER_STOPWORDS = {"No","Description","Unit","MRP/RSP","HSN","Qty","Product","Rate","Disc.",
                    "Taxable","Amt.","CGST","S/UT","GST","Cess","Total","Sl.","Item","&",
                    "SR","MRP","RSP","Price","Net","Amount","Tax","Type","Gross","Discounts",
                    "/Coupons","Value","₹","/UTGST","/UT","GSTIN:","FSSAI:"}

def row_groups(record, spans):
    """Anchor item rows by HSN y-position (Zepto/Flipkart/Amazon tables). Rows:
    name_range = reading-order token interval before the anchor; value tokens are
    assigned by nearest anchor within a y-window (keeps footer subtotals out).
    HSN-less fee rows (e.g. 'Handling Fee') become pseudo-rows anchored on QUANTITY."""
    words = record["words"]
    hsn = [s for s in spans if s[2] == "HSN_SAC"]
    anchors = []  # (span_start_idx, y_center, is_hsn)
    def yc(a, b):
        ys = [record["boxes"][i][1] for i in range(a, b + 1)]
        return sum(ys) / len(ys)
    for a, b, f in hsn:
        anchors.append((a, yc(a, b), True))
    qty = [s for s in spans if s[2] == "QUANTITY"]
    for a, b, f in qty:
        y = yc(a, b)
        if not any(abs(y - ay) < 45 for _, ay, _ in anchors):
            anchors.append((a, y, False))
    if not anchors:
        # HSN-less layout (e.g. Amazon marketplace-fee page): anchor on value fields
        for fname in ("UNIT_PRICE", "TAXABLE_VALUE", "TOTAL"):
            for a, b, f in spans:
                if f == fname:
                    anchors.append((a, yc(a, b), False))
                    break
            if anchors:
                break
    anchors.sort(key=lambda t: t[0])
    rows = []
    for k, (astart, ay, is_hsn) in enumerate(anchors):
        prev_end = anchors[k - 1][0] - 1 if k > 0 else 0
        rows.append({
            "anchor": astart, "y": ay, "is_hsn": is_hsn,
            "name_range": (prev_end, astart - 1),
        })
    return rows

def record_to_invoice(record: dict) -> dict:
    """Convert a BIOES-labeled record into the Tally-ready invoice dict."""
    spans = field_spans(record)
    words, boxes = record["words"], record["boxes"]

    def get(field):
        return [span_text(record, a, b) for a, b, f in spans if f == field]

    seller_name = get("SELLER_NAME")
    seller_gstin = get("SELLER_GSTIN")
    inv_no = get("INVOICE_NUMBER")
    dates = get("INVOICE_DATE")
    pos_names = get("PLACE_OF_SUPPLY")

    # ---- items from rows
    items = []
    rows = row_groups(record, spans)
    # Pre-assign each span to exactly one row: ITEM_NAME by name_range containment,
    # value fields by nearest anchor within the y-window. Spans never double-count.
    span_rows = {}
    for a, b, f in spans:
        y = sum(record["boxes"][i][1] for i in range(a, b + 1)) / (b - a + 1)
        if f == "ITEM_NAME":
            for row in rows:
                lo, hi = row["name_range"]
                if lo <= a and b <= hi:
                    span_rows[(a, b)] = (row, f)
                    break
        if (a, b) not in span_rows:
            best, best_d = None, None
            for row in rows:
                if abs(y - row["y"]) < 45:
                    d = abs(y - row["y"])
                    if best_d is None or d < best_d:
                        best, best_d = row, d
            if best is not None:
                span_rows[(a, b)] = (best, f)
    for row in rows:
        hsn = ""
        desc_parts, qty, rate, taxable, gst_rate, line_total = [], None, None, None, 0.0, None
        disc_pct, disc_abs = None, None
        cgst_amt = sgst_amt = igst_amt = tax_amt = None
        for (a, b), (owner, f) in span_rows.items():
            if owner is not row:
                continue
            txt = span_text(record, a, b)
            if f == "HSN_SAC":
                hsn = re.sub(r"^HSN[:]?", "", txt)
            elif f == "ITEM_NAME":
                parts = txt.split()
                parts = [p for p in parts if p not in HEADER_STOPWORDS]
                if parts:
                    desc_parts.append(" ".join(parts))
            elif f == "QUANTITY":
                qty = int(clean_amount(txt))
            elif f == "UNIT_PRICE":
                rate = clean_amount(txt)
            elif f == "DISCOUNT":
                if "%" in txt:
                    disc_pct = float(re.sub(r"[^0-9.]", "", txt))
                else:
                    disc_abs = clean_amount(txt)
            elif f == "TAXABLE_VALUE":
                taxable = clean_amount(txt)
            elif f in ("CGST_RATE", "SGST_RATE", "IGST_RATE"):
                gst_rate += float(re.sub(r"[^0-9.]", "", txt))
            elif f == "CGST_AMOUNT":
                cgst_amt = clean_amount(txt)
            elif f == "SGST_AMOUNT":
                sgst_amt = clean_amount(txt)
            elif f == "IGST_AMOUNT":
                igst_amt = clean_amount(txt)
            elif f == "TAX_AMOUNT":
                tax_amt = clean_amount(txt)
            elif f == "TOTAL":
                line_total = clean_amount(txt)
        desc = " ".join(desc_parts)[:200]
        if taxable is None and rate is not None and qty is not None:
            taxable = round(rate * qty, 2)
        discount = None
        if disc_pct is not None and rate is not None:
            discount = round(rate * disc_pct / 100.0, 2)
        elif disc_abs is not None:
            discount = disc_abs
        if not hsn and not desc:
            continue
        items.append({
            "hsn": hsn, "desc": desc, "qty": qty or 1, "unit": "Nos",
            "rate": round(rate, 2) if rate is not None else 0.0,
            "discount": discount,
            "taxable": round(taxable, 2) if taxable is not None else 0.0,
            "gst_rate": round(gst_rate, 2),
            "cgst_amt": round(cgst_amt, 2) if cgst_amt is not None else None,
            "sgst_amt": round(sgst_amt, 2) if sgst_amt is not None else None,
            "igst_amt": round(igst_amt, 2) if igst_amt is not None else None,
            "tax_amt": round(tax_amt, 2) if tax_amt is not None else None,
            "total": round(line_total, 2) if line_total is not None else None,
        })

    # ---- totals: grand TOTAL = largest TOTAL span NOT assigned to any item row
    rows = row_groups(record, spans)
    row_totals = set()
    for row in rows:
        for a, b, f in spans:
            if f == "TOTAL":
                if (a, b) in span_rows and span_rows[(a, b)][0] is row:
                    row_totals.add((a, b))
    grand_totals = []
    for a, b, f in spans:
        if f == "TOTAL" and (a, b) not in row_totals:
            grand_totals.append(clean_amount(span_text(record, a, b)))
    total = round(max(grand_totals), 2) if grand_totals else None
    if total is None:
        total = round(sum(it["taxable"] * (1 + it["gst_rate"] / 100) for it in items), 2)

    # ---- place of supply name -> 2-digit state code
    pos_code = ""
    for nm in pos_names:
        norm = STATE_NAME_FIX.get(nm.strip().upper(), nm.strip().upper())
        code = NAME_TO_CODE.get(norm)
        if code:
            pos_code = code
            break
    if not pos_code and seller_gstin:
        pos_code = seller_gstin[0][:2]  # fallback: assume same-state (intra)

    date_raw = dates[0] if dates else ""
    # normalise to dd-mm-yyyy for tally_xml._d()
    date_out = ""
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%Y-%m-%d", "%d-%b-%Y", "%d %B %Y", "%d %b %Y"):
        try:
            date_out = datetime.strptime(date_raw, fmt).strftime("%d-%m-%Y")
            break
        except (ValueError, TypeError):
            continue

    return {
        "gstin_supplier": seller_gstin[0] if seller_gstin else "",
        "invoice_no": inv_no[0] if inv_no else "",
        "date": date_out,
        "place_of_supply": pos_code,
        "party_name": seller_name[0] if seller_name else "",
        "items": items,
        "total": total,
    }


def record_to_import_xml(record: dict, voucher_type: str = "Purchase",
                         company: str = "", include_inventory: bool = False,
                         ledger_map: dict | None = None,
                         dedup: InvoiceDeduplicator | None = None,
                         check_dedup: bool = False,
                         record_dedup: bool = False,
                         force: bool = False) -> tuple[dict, str]:
    """Full pipeline: record -> validated invoice -> Tally import XML."""
    invoice = record_to_invoice(record)
    report = validate_invoice(invoice)
    invoice["info"] = report["info"]

    if check_dedup or record_dedup:
        dedup_engine = dedup or InvoiceDeduplicator()
        check = dedup_engine.check_duplicate(invoice)
        invoice["dedup"] = check
        if check["is_duplicate"] and not force:
            if check["status"] in ("EXACT_DUPLICATE", "FILE_HASH_DUPLICATE"):
                raise ValueError(f"Duplicate invoice rejected: {check['message']}")
        if record_dedup and (not check["is_duplicate"] or force):
            try:
                dedup_engine.record_invoice(invoice)
            except ValueError:
                pass

    xml = build_import(invoice, voucher_type=voucher_type, company=company,
                       include_inventory=include_inventory, ledger_map=ledger_map)
    return invoice, xml


def load_record(path_or_idx, dataset: str | None = None) -> dict:
    if dataset is None:
        dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gst_smoke_dataset.jsonl")
    recs = [json.loads(l) for l in open(dataset)]
    if isinstance(path_or_idx, int):
        return recs[path_or_idx]
    for r in recs:
        if path_or_idx in r["image_id"]:
            return r
    raise KeyError(f"no record matching {path_or_idx!r} in {dataset}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="BIOES record -> Tally import XML")
    ap.add_argument("record", help="image_id substring or 0-based index into the dataset")
    ap.add_argument("--dataset", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gst_smoke_dataset.jsonl"))
    ap.add_argument("--voucher-type", default="Purchase")
    ap.add_argument("--company", default="")
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--out", default=None, help="write XML to this file")
    ap.add_argument("--post", default=None, help="POST to Tally HTTP endpoint e.g. 127.0.0.1:9000")
    ap.add_argument("--no-dedup", action="store_true", help="Disable deduplication check")
    ap.add_argument("--record-dedup", action="store_true", help="Record invoice in deduplication registry")
    ap.add_argument("--force", action="store_true", help="Force import even if duplicate is detected")
    args = ap.parse_args()

    rec = load_record(args.record, args.dataset)
    check_dedup = not args.no_dedup

    try:
        invoice, xml = record_to_import_xml(
            rec,
            voucher_type=args.voucher_type,
            company=args.company,
            include_inventory=args.inventory,
            check_dedup=check_dedup,
            record_dedup=args.record_dedup,
            force=args.force,
        )
    except ValueError as e:
        print(f"\n[DUPLICATE REJECTED] {e}", file=sys.stderr)
        sys.exit(2)

    print(f"record : {rec['image_id']}")
    print(f"invoice: {json.dumps(invoice, indent=2, default=str)}")
    valid = validate_invoice(invoice)
    print(f"valid  : {valid['valid']}  errors={valid['errors']}  warnings={valid['warnings']}")
    print(f"inter-state: {invoice.get('info', {}).get('inter_state')}")
    if "dedup" in invoice:
        print(f"dedup  : {invoice['dedup']['status']} (duplicate={invoice['dedup']['is_duplicate']})")

    if args.out:
        with open(args.out, "w") as f:
            f.write(xml)
        print(f"wrote {args.out} ({len(xml)} bytes)")
    if args.post:
        from tally_client import post_xml, parse_result
        host, _, port = args.post.partition(":")
        resp = post_xml(xml, host=host, port=int(port or 9000))
        print("tally response:", json.dumps(parse_result(resp), indent=2)[:600])
