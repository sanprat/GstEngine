"""
GST Validator — deterministic rule layer for Indian GST invoices.
Phase 0 of the gst-engine. No ML, no network: pure code.

Layers:
  1. GSTIN validation  (checksum via Luhn mod-36 + structure + state codes)
  2. PAN validation    (structure + holder type)
  3. HSN code checks   (format; official-list membership is data-driven)
  4. Invoice math      (line totals, GST split, grand total)
  5. GST slab logic    (rates valid as of 2026; special-rate table)

Usage:
    from gst_validator import validate_invoice
    report = validate_invoice(invoice_dict)   # -> dict with errors/warnings per field

Every check returns machine-readable diagnostics so the phone app / MCP server
can decide auto-accept vs human-review without guessing.
"""

from __future__ import annotations
import re
from datetime import datetime, date

# ---------------------------------------------------------------------------
# GSTIN: Luhn mod-36 over full 15-char string (alphabet 0-9A-Z)
# Verified against python-stdnum's canonical in_.gstin implementation.
# ---------------------------------------------------------------------------

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

STATE_CODES = {
    "01": "Jammu & Kashmir", "02": "Himachal Pradesh", "03": "Punjab",
    "04": "Chandigarh", "05": "Uttarakhand", "06": "Haryana",
    "07": "Delhi", "08": "Rajasthan", "09": "Uttar Pradesh",
    "10": "Bihar", "11": "Sikkim", "12": "Arunachal Pradesh",
    "13": "Nagaland", "14": "Manipur", "15": "Mizoram",
    "16": "Tripura", "17": "Meghalaya", "18": "Assam",
    "19": "West Bengal", "20": "Jharkhand", "21": "Odisha",
    "22": "Chhattisgarh", "23": "Madhya Pradesh", "24": "Gujarat",
    "26": "Dadra & Nagar Haveli and Daman & Diu", "27": "Maharashtra",
    "29": "Karnataka", "30": "Goa", "31": "Lakshadweep",
    "32": "Kerala", "33": "Tamil Nadu", "34": "Puducherry",
    "35": "Andaman & Nicobar Islands", "36": "Telangana",
    "37": "Andhra Pradesh", "38": "Ladakh", "97": "Other Territory",
}

PAN_HOLDER_TYPES = {
    "A": "Association of Persons (AOP)",
    "B": "Body of Individuals (BOI)",
    "C": "Company",
    "D": "Firm",
    "F": "Firm (older series)",
    "G": "Government",
    "H": "HUF",
    "J": "Artificial Juridical Person",
    "K": "Local Authority",
    "L": "Local Authority (older series)",
    "P": "Individual (Proprietor)",
    "T": "Trust (AOP)",
}


def luhn_mod36_checksum(number: str) -> int:
    """Luhn mod-N (N=36) sum over `number`. Valid GSTIN => sum % 36 == 0."""
    values = [ALPHABET.index(c) for c in reversed(number)]
    plain = sum(values[::2])
    doubled = sum((v * 2) // 36 + (v * 2) % 36 for v in values[1::2])
    return (plain + doubled) % 36


def calc_check_char(body14: str) -> str:
    """Compute the 15th check character for a 14-char GSTIN body."""
    ck = luhn_mod36_checksum(body14 + ALPHABET[0])
    return ALPHABET[-ck % 36]


def make_gstin(state: str, pan10: str, entity_num: int = 1) -> str:
    """Construct a valid GSTIN (used by the synthetic data generator)."""
    body = state + pan10 + ALPHABET[entity_num] + "Z"
    return body + calc_check_char(body)


def validate_gstin(gstin: str) -> dict:
    """Full GSTIN validation. Returns {'valid': bool, 'errors': [...], 'info': {...}}."""
    errors = []
    info = {}
    g = (gstin or "").strip().upper().replace(" ", "")

    if not re.fullmatch(r"[0-9A-Z]{15}", g):
        errors.append(f"FORMAT: must be exactly 15 alphanumeric chars, got {len(g)}")
        return {"valid": False, "errors": errors, "info": info}

    state = g[:2]
    if state not in STATE_CODES:
        errors.append(f"STATE: unknown state code '{state}'")
    else:
        info["state"] = STATE_CODES[state]

    pan10 = g[2:12]
    if not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan10):
        errors.append(f"PAN_STRUCTURE: chars 3-12 ('{pan10}') don't match PAN pattern AAAPP9999A")
    else:
        holder = PAN_HOLDER_TYPES.get(pan10[3])
        if holder:
            info["pan_holder_type"] = holder
        else:
            errors.append(f"PAN_HOLDER_TYPE: '{pan10[3]}' is not a valid holder-type letter")

    if g[13] != "Z":
        errors.append(f"ENTITY_Z: position 14 must be 'Z', got '{g[13]}'")

    entity_char = g[12]
    if not (ALPHABET.index(entity_char) < 35 and entity_char != "Z" and not entity_char.isdigit() or entity_char.isdigit()):
        # entity position: 1-9 or A-J per GSTN (A=10th registration, etc.); Z reserved
        if entity_char == "Z":
            errors.append(f"ENTITY: position 13 must not be 'Z'")

    if luhn_mod36_checksum(g) != 0:
        expected = calc_check_char(g[:14])
        errors.append(f"CHECKSUM: check digit '{g[14]}' invalid, expected '{expected}'")
    else:
        info["checksum"] = "pass"

    return {"valid": not errors, "errors": errors, "info": info}


# ---------------------------------------------------------------------------
# HSN: format checks + slab lookup (2026 rates; list membership data-driven)
# ---------------------------------------------------------------------------

# GST slabs in force (2026): 0, 0.25, 1, 3, 5, 12, 18, 28 (+ cess on some)
VALID_GST_RATES = {0.0, 0.25, 1.0, 3.0, 5.0, 12.0, 18.0, 28.0}

HSN_FORMAT = re.compile(r"^[0-9]{4}$|^[0-9]{6}$|^[0-9]{8}$")

# Common HSN→default rate map (excerpt — extended via data file, not hardcode-creep)
HSN_DEFAULT_RATES = {
    "8544": 18.0,   # insulated wire/cable
    "8471": 18.0,   # computers & laptops
    "847130": 18.0, # portable data-processing machines
    "8517": 18.0,   # phones
    "9403": 18.0,   # furniture
    "6109": 5.0,    # t-shirts, knitted
    "1006": 0.0,    # rice (husked)
    "2402": 28.0,   # cigarettes
    "2202": 28.0,   # aerated drinks
    "2701": 5.0,    # coal
    "3004": 12.0,   # medicaments
    "8703": 28.0,   # motor cars
    "7108": 3.0,    # gold
    "7113": 3.0,    # jewellery
}


def validate_hsn(hsn: str, declared_rate: float | None = None) -> dict:
    errors, warnings, info = [], [], {}
    h = (hsn or "").strip().replace(" ", "")
    if not HSN_FORMAT.fullmatch(h):
        errors.append(f"HSN_FORMAT: '{hsn}' must be 4, 6, or 8 digits")
        return {"valid": False, "errors": errors, "warnings": warnings, "info": info}
    info["hsn"] = h

    if len(h) == 4:
        warnings.append("HSN_LENGTH: 4-digit — fine under ₹5cr turnover; 6-digit mandatory above (FY26 norms)")
    # Categories with genuine rate ambiguity (food grains, packaged/unpackaged variants etc.)
    AMBIGUOUS_HSN = {"1006", "0402", "1001", "1101", "0901", "0902", "2402", "2202"}
    if h in HSN_DEFAULT_RATES and declared_rate is not None:
        expected = HSN_DEFAULT_RATES[h]
        if abs(float(declared_rate) - expected) > 0.01:
            if h in AMBIGUOUS_HSN:
                warnings.append(f"HSN_RATE_VARIANT: HSN {h} default {expected}%, invoice declares {declared_rate}% — legit variant category, flag for review")
            else:
                errors.append(f"HSN_RATE_MISMATCH: HSN {h} standard rate {expected}%, invoice declares {declared_rate}%")
        else:
            info["rate_matches_standard"] = True

    return {"valid": not errors, "errors": errors, "warnings": warnings, "info": info}


# ---------------------------------------------------------------------------
# Invoice math
# ---------------------------------------------------------------------------

def _round2(x: float) -> float:
    return round(x + 1e-9, 2)


def validate_invoice_math(invoice: dict, tolerance: float = 0.02) -> dict:
    """
    Validates arithmetic of an invoice dict:
      {items: [{hsn, qty, rate, taxable, gst_rate}], total, place_of_supply, gstin_supplier}
    Inter-intra split: CGST+SGST unless place_of_supply state != supplier state → IGST.
    """
    errors, warnings, info = [], [], {}

    items = invoice.get("items") or []
    if not items:
        errors.append("MATH: no line items")
        return {"valid": False, "errors": errors, "warnings": warnings, "info": info}

    computed_taxable = 0.0
    computed_gst = 0.0
    for i, it in enumerate(items):
        qty = float(it.get("qty", 0) or 0)
        rate = float(it.get("rate", 0) or 0)
        taxable = it.get("taxable")
        gst_rate = float(it.get("gst_rate", 0) or 0)

        discount = float(it.get("discount", 0) or 0)
        if taxable is not None and qty and rate:
            expected = _round2(qty * rate - discount)
            if abs(expected - float(taxable)) > tolerance:
                errors.append(f"MATH: item {i}: qty×rate={expected} but taxable={taxable}")
        taxable_f = float(taxable if taxable is not None else _round2(qty * rate - discount))
        computed_taxable += taxable_f
        computed_gst += _round2(taxable_f * gst_rate / 100.0)

        if gst_rate not in VALID_GST_RATES:
            errors.append(f"SLAB: item {i} gst_rate {gst_rate}% is not a valid GST slab")

    computed_taxable = _round2(computed_taxable)
    computed_gst = _round2(computed_gst)
    info["computed_taxable"] = computed_taxable
    info["computed_gst"] = computed_gst

    total = invoice.get("total")
    if total is not None:
        expected_total = _round2(computed_taxable + computed_gst)
        # Per-line GST rounding accumulates: allow ₹1 per line item for the grand total.
        total_tol = max(tolerance, 1.0 * len(items))
        if abs(expected_total - float(total)) > total_tol:
            errors.append(f"MATH: total={total} but taxable({computed_taxable})+GST({computed_gst})={expected_total}")
        else:
            info["total_matches"] = True

    # Intra/inter-state split
    sup = validate_gstin(invoice.get("gstin_supplier", ""))
    pos_state = str(invoice.get("place_of_supply", ""))[:2]
    if sup["valid"] and pos_state in STATE_CODES:
        sup_state = invoice.get("gstin_supplier", "")[:2]
        inter = sup_state != pos_state
        info["inter_state"] = inter
        info["expected_split"] = "IGST" if inter else "CGST+SGST"
        gst_split = invoice.get("gst_split")  # {'igst': x} or {'cgst': x, 'sgst': x}
        if gst_split:
            if not inter and "igst" in gst_split and float(gst_split.get("igst", 0) or 0) > 0:
                errors.append("SPLIT_TYPE: intra-state supply but IGST declared — should be CGST+SGST")
            if inter and ("cgst" in gst_split or "sgst" in gst_split):
                errors.append("SPLIT_TYPE: inter-state supply but CGST/SGST declared — should be IGST")
            if inter:
                ig = float(gst_split.get("igst", 0) or 0)
                if abs(ig - computed_gst) > tolerance:
                    errors.append(f"IGST: computed {computed_gst} but invoice IGST={ig}")
            else:
                cg = float(gst_split.get("cgst", 0) or 0)
                sg = float(gst_split.get("sgst", 0) or 0)
                if abs(_round2(cg + sg) - computed_gst) > tolerance:
                    errors.append(f"CGST+SGST: computed {computed_gst} but invoice has {cg}+{sg}={_round2(cg+sg)}")
                if abs(cg - sg) > tolerance:
                    errors.append(f"CGST≠SGST: {cg} vs {sg} (must be equal intra-state)")
    else:
        warnings.append("SPLIT: supplier GSTIN or place_of_supply missing — cannot verify IGST/CGST-SGST split")

    return {"valid": not errors, "errors": errors, "warnings": warnings, "info": info}


# ---------------------------------------------------------------------------
# Invoice-level: duplicate heuristics + date sanity + fraud flags
# ---------------------------------------------------------------------------

DATE_FORMATS = ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%b-%Y", "%d-%m-%y")


def parse_invoice_date(s: str):
    s = (s or "").strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def validate_invoice(invoice: dict, seen_ids: set | None = None) -> dict:
    """
    Master validation. `seen_ids` = set of previously seen invoice numbers
    (per supplier) for duplicate detection.
    Returns {'valid', 'errors', 'warnings', 'info', 'fields': {field: report}}.
    """
    errors, warnings, info, fields = [], [], {}, {}

    # GSTIN supplier (required)
    g = invoice.get("gstin_supplier")
    if not g:
        errors.append("REQUIRED: gstin_supplier missing")
    else:
        rep = validate_gstin(g)
        fields["gstin_supplier"] = rep
        if not rep["valid"]:
            errors.extend(f"SUPPLIER_GSTIN {e}" for e in rep["errors"])

    # Buyer GSTIN (optional but checked if present)
    gb = invoice.get("gstin_buyer")
    if gb:
        rep = validate_gstin(gb)
        fields["gstin_buyer"] = rep
        if not rep["valid"]:
            warnings.extend(f"BUYER_GSTIN {e}" for e in rep["errors"])

    # Supplier-buyer state conflict warning
    if g and gb and validate_gstin(g)["valid"] and validate_gstin(gb)["valid"] and g[:2] == gb[:2]:
        info["intra_state_parties"] = True

    # Invoice number & duplicates
    inv_no = (invoice.get("invoice_no") or "").strip()
    if not inv_no:
        errors.append("REQUIRED: invoice_no missing")
    elif seen_ids and (g, inv_no) in seen_ids:
        errors.append(f"DUPLICATE: invoice '{inv_no}' from this supplier already recorded")

    # Date sanity
    d = parse_invoice_date(invoice.get("date", ""))
    if d is None:
        errors.append(f"DATE: unparseable '{invoice.get('date')}'")
    else:
        info["date"] = d.isoformat()
        if d > date.today():
            warnings.append(f"DATE_FUTURE: invoice dated in the future ({d})")
        if d.year < 2017:
            warnings.append(f"DATE_OLD: pre-GST era date ({d})")

    # Per-item HSN + rate
    for i, it in enumerate(invoice.get("items", [])):
        hrep = validate_hsn(it.get("hsn"), it.get("gst_rate"))
        if not hrep["valid"]:
            errors.extend(f"ITEM{i} {e}" for e in hrep["errors"])
        warnings.extend(f"ITEM{i} {w}" for w in hrep["warnings"])

    # Math
    mrep = validate_invoice_math(invoice)
    fields["math"] = mrep
    errors.extend(mrep["errors"])
    warnings.extend(mrep["warnings"])
    info.update(mrep["info"])

    # Fraud heuristics
    total = invoice.get("total")
    if total is not None and float(total) % 1 == 0 and float(total) >= 10000:
        warnings.append("ROUND_NUMBER: total is a whole ≥10k figure — check for manipulated invoice")

    valid = not errors
    return {"valid": valid, "errors": errors, "warnings": warnings, "info": info, "fields": fields}


if __name__ == "__main__":
    # Self-test with the round-trip-verified GSTINs
    demo = {
        "gstin_supplier": "27ABCTY1234E1ZL",
        "gstin_buyer": "29AXTPS4521K1ZO",
        "invoice_no": "ST/2026/0417",
        "date": "12-10-2026",
        "place_of_supply": "29",
        "items": [
            {"hsn": "8544", "desc": "Copper wire 1.5mm", "qty": 200, "rate": 142.0,
             "taxable": 28400.0, "gst_rate": 18.0},
        ],
        "gst_split": {"igst": 5112.0},
        "total": 33512.0,
    }
    import json
    print(json.dumps(validate_invoice(demo), indent=2, default=str))
