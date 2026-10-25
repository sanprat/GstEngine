"""
Tally HTTP client — POSTs import envelopes to TallyPrime's :9000 and parses results.
Phase 0 of gst-engine.

High-level: import_invoice(invoice, host, port) handles the full dance:
  1. POST envelope
  2. Parse CREATED / ALTERED / IGNORED / LINEERROR
  3. On "Ledger ... does not exist" -> POST ledger-master, then retry once
Idempotent by REMOTEID (safe to retry after network drops).
"""

from __future__ import annotations
import urllib.request
import urllib.error
from tally_xml import build_import
from gst_validator import validate_invoice


def post_xml(xml: str, host: str = "127.0.0.1", port: int = 9000, timeout: int = 30) -> str:
    url = f"http://{host}:{port}"
    req = urllib.request.Request(
        url, data=xml.encode("utf-8"),
        headers={"Content-Type": "text/xml;charset=utf-8",
                 "User-Agent": "gst-engine/0.1"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def parse_result(response_xml: str) -> dict:
    """Extract CREATED/ALTERED/IGNORED/LINEERROR outcomes."""
    import re
    out = {"created": [], "altered": [], "ignored": [], "errors": []}
    for kind, pattern in (("created", r"<CREATED>(.*?)</CREATED>"),
                          ("altered", r"<ALTERED>(.*?)</ALTERED>"),
                          ("ignored", r"<IGNORED>(.*?)</IGNORED>"),
                          ("errors", r"<LINEERROR>(.*?)</LINEERROR>")):
        out[kind].extend(re.findall(pattern, response_xml, re.S))
    return out


def import_invoice(invoice: dict, host: str = "127.0.0.1", port: int = 9000,
                   voucher_type: str = "Purchase", company: str = "",
                   validate: bool = True, ledger_map: dict | None = None) -> dict:
    """
    Validate -> build -> POST -> auto-fix missing ledger -> retry.
    Refuses to import invoices that fail validation (unless validate=False).
    """
    if validate:
        report = validate_invoice(invoice)
        if not report["valid"]:
            return {"imported": False, "reason": "validation_failed",
                    "validation": report}

    xml = build_import(invoice, voucher_type=voucher_type, company=company,
                       ledger_map=ledger_map)
    resp = post_xml(xml, host, port)
    result = parse_result(resp)

    # Missing-ledger recovery: create ALL referenced ledgers, then retry once.
    if result["errors"] and any("does not exist" in e for e in result["errors"]):
        from tally_xml import envelope, ledger_master_xml, voucher_xml
        from gst_validator import validate_gstin
        import re as _re
        sup_state = str(invoice.get("gstin_supplier", ""))[:2]
        pos = str(invoice.get("place_of_supply", ""))[:2]
        is_inter = bool(sup_state and pos and sup_state != pos)
        tax_ledgers = ["Purchase Accounts", "IGST"] if is_inter else ["Purchase Accounts", "CGST", "SGST"]
        party = str(invoice.get("party_name", invoice.get("gstin_supplier", "")))
        for name in [party] + tax_ledgers:
            parent = "Sundry Creditors" if name == party else "Duties & Taxes"
            gstin = invoice.get("gstin_supplier", "") if name == party else ""
            post_xml(envelope(ledger_master_xml(name, parent, gstin), company), host, port)
        retry_xml = envelope(voucher_xml(invoice, voucher_type, ledger_map=ledger_map), company)
        resp = post_xml(retry_xml, host, port)
        result = parse_result(resp)
        result["recovered_missing_ledger"] = True

    imported = bool(result["created"] or result["altered"]) and not result["errors"]
    return {"imported": imported, "result": result, "raw": resp}


if __name__ == "__main__":
    demo = {
        "gstin_supplier": "27ABCTY1234E1ZL",
        "invoice_no": "ST/2026/0417",
        "date": "12-10-2026",
        "place_of_supply": "29",
        "party_name": "Sharma Traders",
        "items": [{"hsn": "8544", "desc": "Copper wire 1.5mm", "qty": 200,
                   "rate": 142.0, "taxable": 28400.0, "gst_rate": 18.0}],
        "total": 33512.0,
    }
    from gst_validator import validate_invoice
    demo["info"] = validate_invoice(demo)["info"]
    import json
    print(json.dumps(import_invoice(demo), indent=2))
