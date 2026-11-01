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

# normalise state names ("MAHARASTRA" typo on a real retail invoice -> Maharashtra)
STATE_NAME_FIX = {
    "MAHARASTRA": "MAHARASHTRA", "MAHARASHTRA": "MAHARASHTRA",
    "KARNATAKA": "KARNATAKA", "DELHI": "DELHI", "MUMBAI": "MAHARASHTRA",
    "TAMIL NADU": "TAMIL NADU", "TELANGANA": "TELANGANA", "KERALA": "KERALA",
    "GOA": "GOA", "UTTAR PRADESH": "UTTAR PRADESH", "RAJASTHAN": "RAJASTHAN",
}
NAME_TO_CODE = {v.upper(): k for k, v in STATE_CODES.items()}

def field_spans(record):
    """Yield (field, start_i, end_i) spans from BIOES labels, in reading order.

    Proper state machine: S- gives a single-token span; B- opens a span that
    consumes following I-/E- tokens of the SAME field; a lone B- with no
    following E- (single-word entity, e.g. OCR-merged multi-word seller label) still closes as [i, i]; stray
    I-/E- without a B- are skipped (malformed predictions must not corrupt).
    """
    spans = []
    labels = record["labels"]
    n = len(labels)
    i = 0
    while i < n:
        l = labels[i]
        if l == "O":
            i += 1
            continue
        pre, _, field = l.partition("-")
        if pre == "S":
            spans.append((i, i, field))
            i += 1
            continue
        if pre == "B":
            j = i + 1
            while j < n:
                pj, _, fj = labels[j].partition("-")
                if fj != field or pj not in ("I", "E"):
                    break
                if pj == "E":
                    j += 1
                    break
                j += 1
            spans.append((i, j - 1, field))
            i = j
            continue
        # stray I-/E- without B-: skip token
        i += 1
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
    """Anchor item rows by HSN y-position (e-commerce tables). Rows:
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
        # HSN-less layout (e.g. marketplace-fee page): anchor on value fields
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

# ---- Field-level post-processing guards (v10 fix, Nov 2026) ----
# The v9 model occasionally labels prominent document titles as BUYER_NAME
# (e.g. "TAX INVOICE" on retail layouts it never saw), leaves invoice numbers
# missing on letter-prefixed bill numbers ("BI2608..."), and can pick a
# PLACE_OF_SUPPLY state that contradicts the seller GSTIN state code. These
# helpers are pure post-processing: reject boilerplate, regex-rescue fields the
# model missed, and flag state-code conflicts instead of silently trusting.

BUYER_BOILERPLATE = {
    "TAX INVOICE", "INVOICE", "GST INVOICE", "GST TAX INVOICE", "CASH MEMO",
    "BILL", "BILL TO", "BILLED TO", "DELIVERY NOTE", "DELIVERY CHALLAN",
    "DEBIT NOTE", "CREDIT NOTE", "PURCHASE ORDER", "SALES INVOICE",
    "ORIGINAL FOR RECIPIENT", "FOR RECIPIENT", "DUPLICATE FOR TRANSPORTER",
    "DUPLICATE FOR SUPPLIER", "ORIGINAL FOR TRANSPORTER", "STATEMENT",
    "RECEIPT", "TAX INVOICE/", "INVOICE/", "SAMPLE", "SAMPLE INVOICE",
    "PROFORMA INVOICE", "ESTIMATE", "QUOTATION", "SUPPLIER LEDGER",
    "PURCHASE ACCOUNTS", "PURCHASE LEDGER", "SALES RETURN", "CREDIT MEMO",
    "INVOICE NO", "INVOICE NUMBER", "INV NO", "BILL NO", "BILL NUMBER",
    "DATE", "TIME", "ORDER ID", "CUSTOMER GST", "COMPANY GST", "GSTIN",
    "IRN DETAILS", "IRN NO", "IRN", "PLACE OF SUPPLY", "BILL FROM",
    "PAYMENT MODE", "PAYMENT SUMMARY", "TAX SUMMARY", "PRODUCTS",
}
BUYER_BOILERPLATE_SUBSTR = ("GSTIN", "TIN:", "PAN:", "CIN:", "FSSAI", "HSN",
                            "SAC", "CGST", "SGST", "IGST", "TOTAL", "PHONE",
                            "EMAIL", "WWW.", "HTTP", "PLACE OF SUPPLY",
                            "TAXABLE", "AMOUNT", "QTY", "RATE", "INVOICE NO",
                            "INVOICE NUMBER", "BILL NO", "ORDER ID", "MOB",
                            "MUMBAI", "MAHARASHTRA", "KARNATAKA", "DELHI",
                            "ADDRESS", "PIN", "PAYMENT", "GST", "TAX SUMMARY",
                            "IRN", "DATE", "TIME", "SHIPPING", "WRAPPING",
                            "DONATION", "SERIAL NO", "ITEM CODE", "DESCRIPTION")
_BUYER_LABEL_RE = re.compile(
    r"(?:Billed\s*To|Bill\s*To|Customer(?:\s*Name)?|Buyer(?:\s*Name)?|"
    r"Recipient(?:\s*Name)?|Party(?:\s*Name)?|Purchaser(?:\s*Name)?|"
    r"Consignee|Client(?:\s*Name)?)\s*[:.\-]?\s*([A-Z][A-Za-z.\s]{2,40})",
    re.IGNORECASE)
_INV_LABEL_RE = re.compile(
    r"(?:Invoice|Inv\.?|Bill|Tax\s*Invoice|GST\s*Invoice)\s*(?:No|Number|#)?"
    r"\.?[:#\s-]+([A-Za-z0-9][A-Za-z0-9\-/]{3,24})", re.IGNORECASE)
_INV_CORP_RE = re.compile(r"\b(\d{4,6}[A-Z]{2}\d{6,12})\b")
_INV_ALPHA_RE = re.compile(r"\b([A-Z]{2,4}\d{6,14})\b")


_GSTIN_STRICT = r"\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]"


def find_valid_gstin(text: str) -> str:
    """OCR-tolerant GSTIN search: find 15-char alnum candidates in raw OCR text,
    try ambiguous-glyph corrections (1<->Z, 0<->O, I<->1...) and return the
    first Mod-36-valid one. Retail OCR routinely corrupts a real GSTIN
    into the OCR-garbled GSTIN — the strict regex alone will never match that.
    """
    from gst_validator import validate_gstin
    text = (text or "").upper()
    for m in re.finditer(r"(?<![A-Z0-9])([A-Z0-9]{15})(?![A-Z0-9])", text):
        cand = m.group(1)
        corr = correct_gstin_ocr(cand)
        if validate_gstin(corr).get("valid"):
            return corr
    return ""


def correct_gstin_ocr(gstin: str) -> str:
    """Repair OCR-corrupted GSTINs by swapping ambiguous glyphs (1<->Z, 0<->O,
    8<->B, I<->1, S<->5, 2<->Z) one position at a time until Mod-36 validates.

    Retail invoices (big-box retail layout) routinely OCR-rotate '1' as 'Z' in GSTINs
    the OCR-garbled GSTIN (1/Z swap) — the model can only output the characters
    present in its input, so the correction happens in post-processing.
    """
    from gst_validator import validate_gstin
    g = (gstin or "").strip().upper()
    if len(g) != 15 or validate_gstin(g)["valid"]:
        return g
    AMBIG = {"0": "O", "O": "0", "1": "Z", "Z": "1", "I": "1", "8": "B",
             "B": "8", "S": "5", "5": "S", "2": "Z"}
    seen = {g}
    stack = [g]
    while stack:
        cand = stack.pop()
        for i, ch in enumerate(cand):
            alt = AMBIG.get(ch)
            if not alt or alt == ch:
                continue
            new = cand[:i] + alt + cand[i + 1:]
            if new in seen:
                continue
            seen.add(new)
            if validate_gstin(new)["valid"]:
                return new
            stack.append(new)
    return g


def is_buyer_boilerplate(text: str) -> bool:
    """True if a BUYER_NAME candidate is actually a document title/boilerplate."""
    t = " ".join(str(text).upper().split())
    if not t:
        return True
    if t in BUYER_BOILERPLATE:
        return True
    if any(s in t for s in BUYER_BOILERPLATE_SUBSTR):
        return True
    # Pure punctuation / too short / all-caps title token like "TAX INVOICE" repeated
    if len(t) < 3:
        return True
    return False


def buyer_regex_fallback(words, boxes=None) -> str:
    """Regex-rescue buyer while respecting the OCR visual line."""
    words = [str(w) for w in words]
    if boxes and len(boxes) == len(words):
        labels = {"billed", "bill", "customer", "buyer", "recipient", "party",
                  "purchaser", "consignee", "client"}
        for i, word in enumerate(words):
            if word.strip(" ,:.-").lower() not in labels:
                continue
            label_y = (boxes[i][1] + boxes[i][3]) / 2
            j = i + 1
            # Consume label suffixes: To, Name, Party, colon punctuation.
            while j < len(words) and words[j].strip(" ,:.-").lower() in {
                "to", "name", "party", "recipient", "customer", ":"
            }:
                j += 1
            parts = []
            for k in range(j, min(len(words), j + 6)):
                token = words[k].strip(" ,:-")
                if not token or any(c.isdigit() for c in token) or "@" in token:
                    break
                y = (boxes[k][1] + boxes[k][3]) / 2
                if abs(y - label_y) > 14:
                    break
                if is_buyer_boilerplate(token):
                    break
                if not re.fullmatch(r"[A-Za-z.]+", token):
                    break
                parts.append(token)
            if parts:
                cand = " ".join(parts)
                if not is_buyer_boilerplate(cand) and len(cand) >= 3:
                    return re.sub(r"([A-Za-z])\1{2,}", r"\1\1", cand)
        return ""

    # Text-only fallback for callers without geometry.
    text = " ".join(words)
    m = _BUYER_LABEL_RE.search(text)
    if not m:
        return ""
    cand = m.group(1).strip(" ,:-")
    if is_buyer_boilerplate(cand) or len(cand) < 3:
        return ""
    return re.sub(r"([A-Za-z])\1{2,}", r"\1\1", cand)


def invoice_no_regex_fallback(words) -> str:
    """Regex-rescue invoice number from OCR words (corporate + letter-prefixed)."""
    text = " ".join(str(w) for w in words)
    for pat in (_INV_CORP_RE, _INV_LABEL_RE, _INV_ALPHA_RE):
        m = pat.search(text)
        if m:
            cand = m.group(1).strip()
            # Reject obvious non-invoice tokens
            low = cand.lower()
            if any(k in low for k in ("gstin", "hsn", "sac", "cgst", "sgst",
                                      "igst", "total", "qty", "rate", "date",
                                      "invoice", "bill", "tax")):
                continue
            if any(ch.isdigit() for ch in cand) and len(cand) >= 4:
                return cand
    return ""


_INV_NOISE = re.compile(
    r"^(TOTAL|INVOICE|INVOICE NO|INVOICE NUMBER|BILL NO|BILL|AMOUNT|QTY|RATE|"
    r"PRODUCTS|DESCRIPTION|TAX INVOICE|GST INVOICE|DATE|SL\s*NO|SERIAL\s*NO|"
    r"SR\.?\s*NO|HSN|SAC|CGST|SGST|IGST|TAXABLE|NIL|SUBTOTAL|GRAND\s*TOTAL|"
    r"TOTAL\s*AMOUNT)$",
    re.IGNORECASE)


def is_inv_noise(text: str) -> bool:
    """True if a model INVOICE_NUMBER span is actually a table header/footer
    word (e.g. 'Total', 'DATE') that page-level extraction picks up."""
    t = str(text or "").strip()
    if not t:
        return True
    if _INV_NOISE.match(t):
        return True
    # Pure 4-digit values are years, not invoice numbers
    if re.fullmatch(r"\d{4}", t):
        return True
    return False


def record_to_invoice(record: dict) -> dict:
    """Convert a BIOES-labeled record into the Tally-ready invoice dict."""
    spans = field_spans(record)
    words, boxes = record["words"], record["boxes"]

    def get(field):
        return [span_text(record, a, b) for a, b, f in spans if f == field]

    seller_name = get("SELLER_NAME")
    seller_gstin = get("SELLER_GSTIN")
    inv_no = [n for n in get("INVOICE_NUMBER") if not is_inv_noise(n)]
    dates = get("INVOICE_DATE")
    pos_names = get("PLACE_OF_SUPPLY")
    # Buyer: take the FIRST span that is not document-title/address boilerplate.
    # v10 model can emit B-BUYER_NAME on "Invoice No:" AND S-BUYER_NAME on the
    # real name (example data) — naive [0] picked the wrong one.
    buyer_name = [b for b in get("BUYER_NAME") if not is_buyer_boilerplate(b)]

    warnings = []
    # Buyer fallback: model said nothing clean OR everything was boilerplate
    if not buyer_name:
        fb = buyer_regex_fallback(words, boxes)
        if fb:
            buyer_name = [fb]
            warnings.append("BUYER_NAME recovered by regex fallback")
    # Invoice number fallback: model missed letter-prefixed bill numbers (BI2608..)
    if not inv_no:
        fb_inv = invoice_no_regex_fallback(words)
        if fb_inv:
            inv_no = [fb_inv]
            warnings.append("INVOICE_NUMBER recovered by regex fallback")
    # Seller GSTIN: accept the model's span ONLY if it looks like a GSTIN;
    # otherwise OCR-tolerant search over the raw words (handles 'IRN DETAILS:'
    # mislabels AND OCR-corrupted the OCR-garbled GSTIN -> real GSTIN).
    raw_all = " ".join(str(w) for w in words)
    if seller_gstin and re.fullmatch(_GSTIN_STRICT, seller_gstin[0].upper()):
        pass  # model span is GSTIN-shaped; use it
    else:
        if seller_gstin:
            warnings.append(f"SELLER_GSTIN span {seller_gstin[0]!r} not GSTIN-shaped; re-searching OCR")
            seller_gstin = []
        fb = find_valid_gstin(raw_all)
        if fb:
            seller_gstin = [fb]
            warnings.append("SELLER_GSTIN recovered by OCR-tolerant GSTIN search")
    if seller_gstin:
        corr = correct_gstin_ocr(seller_gstin[0])
        if corr != seller_gstin[0]:
            seller_gstin = [corr]
            warnings.append(f"SELLER_GSTIN OCR-corrected {seller_gstin[0]} -> {corr}")

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
                if abs(y - row["y"]) < 150:
                    d = abs(y - row["y"])
                    if best_d is None or d < best_d:
                        best, best_d = row, d
            if best is not None:
                span_rows[(a, b)] = (best, f)
    for row in rows:
        hsn = ""
        desc_parts, qty, rate, taxable, line_total = [], None, None, None, None
        disc_pct, disc_abs = None, None
        cgst_rate = sgst_rate = igst_rate = 0.0
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
                # Reject serial numbers/barcodes accidentally labeled as quantity;
                # valid invoice quantities are short numeric tokens (1, 2, 10...).
                if re.fullmatch(r"\s*\d{1,4}(?:\.\d+)?\s*", txt):
                    qty = int(clean_amount(txt))
            elif f == "UNIT_PRICE":
                if rate is None:
                    rate = clean_amount(txt)
            elif f == "DISCOUNT":
                if "%" in txt:
                    disc_pct = float(re.sub(r"[^0-9.]", "", txt))
                else:
                    disc_abs = clean_amount(txt)
            elif f == "TAXABLE_VALUE":
                # Keep the first line-item taxable value; a later 484.03 is
                # the grand total column mislabeled as TAXABLE_VALUE.
                if taxable is None:
                    taxable = clean_amount(txt)
            elif f in ("CGST_RATE", "SGST_RATE", "IGST_RATE"):
                if f == "CGST_RATE":
                    cgst_rate += float(re.sub(r"[^0-9.]", "", txt))
                elif f == "SGST_RATE":
                    sgst_rate += float(re.sub(r"[^0-9.]", "", txt))
                else:
                    igst_rate += float(re.sub(r"[^0-9.]", "", txt))
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
        gst_rate = cgst_rate + sgst_rate + igst_rate
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
        # v10 tax fill (in-loop: cgst_rate/sgst_rate/igst_rate are row-scoped):
        # retail layouts yield taxable + rates but miss split AMOUNTS.
        # Compute the intra-state split when intra rates were found and no
        # amounts exist. Never overwrite model amounts, never invent CGST/SGST
        # when only IGST was charged.
        if (taxable and gst_rate and cgst_amt is None and sgst_amt is None
                and igst_amt is None):
            if cgst_rate or sgst_rate:  # intra-state: CGST + SGST split
                half = gst_rate / 2.0
                items[-1]["cgst_amt"] = round(taxable * half / 100.0, 2)
                items[-1]["sgst_amt"] = round(taxable * half / 100.0, 2)
            elif igst_rate:  # inter-state: full IGST
                items[-1]["igst_amt"] = round(taxable * igst_rate / 100.0, 2)
        # Reverse fill: amounts found but NO rate spans (retail labels CGST/
        # SGST amounts, not rates) -> infer gst_rate from amounts for Tally.
        elif (taxable and gst_rate == 0 and (cgst_amt or sgst_amt)):
            tax_amt = (cgst_amt or 0.0) + (sgst_amt or 0.0)
            items[-1]["gst_rate"] = round(tax_amt / taxable * 100.0, 2)

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

    # Page-level tax summary (bottom block): sum CGST/SGST/IGST amounts
    # NOT attached to any item row. Plausibility guard: a tax span whose value
    # is >= 40% of total taxable is the mislabeled grand-total column
    # (e.g. 484.03 labeled CGST) — GST max rate is 28%, so taxes are always
    # far below 40% of the taxable value. This avoids needing `total` here.
    taxable_total = sum(it["taxable"] for it in items)
    cgst_total = sgst_total = igst_total = 0.0
    for a, b, f in spans:
        if f in ("CGST_AMOUNT", "SGST_AMOUNT", "IGST_AMOUNT") and (a, b) not in span_rows:
            v = clean_amount(span_text(record, a, b))
            if v <= 0:
                continue
            if taxable_total and v >= 0.4 * taxable_total:
                continue  # grand-total column value mislabeled as a tax amount
            if f == "CGST_AMOUNT":
                cgst_total += v
            elif f == "SGST_AMOUNT":
                sgst_total += v
            else:
                igst_total += v

    # Grand total resolution: prefer an explicit TOTAL span; else derive from
    # items (taxable + all tax amounts, incl. page-level tax summary).
    total = round(max(grand_totals), 2) if grand_totals else None
    if (total is None or total <= 0) and items:
        total = round(taxable_total + cgst_total + sgst_total + igst_total, 2)

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

    # Numeric-date fallback: model missed INVOICE_DATE but the OCR text has
    # dd/mm/yyyy ('23/08/2026'). Skip totals/gstin numbers via context.
    if not date_out:
        m_num = re.search(
            r"\b(0?[1-9]|[12][0-9]|3[01])[/\-.]\s?(0?[1-9]|1[0-2])[/\-.]\s?(20[2-9][0-9])\b",
            raw_all)
        if m_num:
            ds = f"{int(m_num.group(1)):02d}-{int(m_num.group(2)):02d}-{m_num.group(3)}"
            date_out = ds
            warnings.append("INVOICE_DATE recovered by numeric-date fallback")

    # POS sanity: GSTIN state code vs extracted place-of-supply state — flag conflict
    if pos_code and seller_gstin and seller_gstin[0][:2] != pos_code:
        warnings.append(
            f"PLACE_OF_SUPPLY state {pos_code} != seller GSTIN state {seller_gstin[0][:2]}; "
            "verify interstate vs intrastate before posting")

    return {
        "gstin_supplier": seller_gstin[0] if seller_gstin else "",
        "invoice_no": inv_no[0] if inv_no else "",
        "date": date_out,
        "place_of_supply": pos_code,
        "party_name": seller_name[0] if seller_name else "",
        "buyer_name": buyer_name[0] if buyer_name else "",
        "items": items,
        "total": total,
        "cgst_total": round(cgst_total, 2) if cgst_total else 0.0,
        "sgst_total": round(sgst_total, 2) if sgst_total else 0.0,
        "igst_total": round(igst_total, 2) if igst_total else 0.0,
        "warnings": warnings,
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
