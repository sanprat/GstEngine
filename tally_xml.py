"""
Tally XML generator — converts validated invoice JSON into TallyPrime import XML.
Phase 0 of gst-engine.

Generates:
  - Purchase/Sales vouchers with inventory entries (optional)
  - Ledger masters auto-creation requests (for missing parties/tax ledgers)
  - REMOTEID on every message for idempotent re-imports

Envelope format per TallyPrime's documented Import Data interface.
"""

from __future__ import annotations
import re
from datetime import datetime
from xml.sax.saxutils import escape


def _d(date_str: str) -> str:
    """Tally XML date format: YYYYMMDD."""
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%b-%Y", "%d-%m-%y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    raise ValueError(f"unparseable date: {date_str!r}")


def _v(x) -> str:
    """Format a number for Tally XML (no thousand separators, 2dp)."""
    return f"{float(x):.2f}"


def voucher_xml(invoice: dict, voucher_type: str = "Purchase", include_inventory: bool = False,
                ledger_map: dict | None = None) -> str:
    """
    Build one TALLYMESSAGE for a voucher from a validated invoice dict.

    ledger_map: optional mapping of invoice party names -> firm's Tally ledger names.
                (The chart-of-accounts problem: 'Sharma Traders' may be 'SHARMA TRADERS MH'.)
    """
    lm = ledger_map or {}
    inv_no = escape(str(invoice.get("invoice_no", "")))
    sup_key = escape(re.sub(r"[^A-Za-z0-9]", "", str(invoice.get("gstin_supplier") or invoice.get("party_name") or "SUPPLIER")).upper())
    remote_id = f"GSTENGINE-{sup_key}-{inv_no}"            # cross-vendor idempotency key
    date_s = _d(str(invoice.get("date")))
    narration = escape(f"GST Engine auto-import {inv_no} from invoice scan")
    party_raw = str(invoice.get("party_name", invoice.get("gstin_supplier", "Sundry Creditors")))
    party = escape(lm.get(party_raw, party_raw))

    is_inter = invoice.get("info", {}).get("inter_state") if "info" in invoice else None
    if is_inter is None:
        sup_state = str(invoice.get("gstin_supplier", ""))[:2]
        pos = str(invoice.get("place_of_supply", ""))[:2]
        is_inter = bool(sup_state and pos and sup_state != pos)

    parts = [f'<TALLYMESSAGE xmlns:UDF="TallyUDF">',
             f'<VOUCHER REMOTEID="{remote_id}" VCHTYPE="{escape(voucher_type)}" ACTION="Create" OBJVIEW="Accounting Voucher View">',
             f"<DATE>{date_s}</DATE>",
             f"<EFFECTIVEDATE>{date_s}</EFFECTIVEDATE>",
             f"<VOUCHERTYPENAME>{escape(voucher_type)}</VOUCHERTYPENAME>",
             f"<VOUCHERNUMBER>{inv_no}</VOUCHERNUMBER>",
             f"<NARRATION>{narration}</NARRATION>",
             "<PERSISTEDVIEW>Accounting Voucher View</PERSISTEDVIEW>"]

    entries = []
    for it in invoice.get("items", []):
        taxable = float(it.get("taxable") or (float(it.get("qty", 0)) * float(it.get("rate", 0))))
        gst_rate = float(it.get("gst_rate", 0))
        gst_amt = round(taxable * gst_rate / 100.0, 2)
        # prefer verbatim extracted tax amounts over rate-computed (rounding fidelity)
        igst_amt = it.get("igst_amt")
        cgst_amt = it.get("cgst_amt")
        sgst_amt = it.get("sgst_amt")
        total = round(taxable + gst_amt, 2)
        if is_inter and igst_amt is not None:
            total = round(taxable + float(igst_amt), 2)
        elif not is_inter and cgst_amt is not None and sgst_amt is not None:
            total = round(taxable + float(cgst_amt) + float(sgst_amt), 2)

        # Party line: credit for Purchase (we owe them), amount negative in Tally convention
        entries.append("<ALLLEDGERENTRIES.LIST>")
        entries.append(f"<LEDGERNAME>{party}</LEDGERNAME>")
        entries.append("<ISDEEMEDPOSITIVE>Yes</ISDEEMEDPOSITIVE>")
        entries.append(f"<AMOUNT>-{_v(total)}</AMOUNT>")
        entries.append("</ALLLEDGERENTRIES.LIST>")

        # Item taxable line
        ledger = escape(str(it.get("purchase_ledger", "Purchase Accounts")))
        entries.append("<ALLLEDGERENTRIES.LIST>")
        entries.append(f"<LEDGERNAME>{ledger}</LEDGERNAME>")
        entries.append("<ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE>")
        entries.append(f"<AMOUNT>{_v(taxable)}</AMOUNT>")
        entries.append("</ALLLEDGERENTRIES.LIST>")

        # Tax ledger(s)
        if is_inter:
            tax_name = escape(str(it.get("igst_ledger", "IGST")))
            tax_amt = float(igst_amt) if igst_amt is not None else gst_amt
            entries.append("<ALLLEDGERENTRIES.LIST>")
            entries.append(f"<LEDGERNAME>{tax_name}</LEDGERNAME>")
            entries.append("<ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE>")
            entries.append(f"<AMOUNT>{_v(tax_amt)}</AMOUNT>")
            entries.append("</ALLLEDGERENTRIES.LIST>")
        else:
            cg = escape(str(it.get("cgst_ledger", "CGST")))
            sg = escape(str(it.get("sgst_ledger", "SGST")))
            cg_v = float(cgst_amt) if cgst_amt is not None else round(gst_amt / 2.0, 2)
            sg_v = float(sgst_amt) if sgst_amt is not None else round(gst_amt - cg_v, 2)
            for name, amt in ((cg, cg_v), (sg, sg_v)):
                entries.append("<ALLLEDGERENTRIES.LIST>")
                entries.append(f"<LEDGERNAME>{name}</LEDGERNAME>")
                entries.append("<ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE>")
                entries.append(f"<AMOUNT>{_v(amt)}</AMOUNT>")
                entries.append("</ALLLEDGERENTRIES.LIST>")

        if include_inventory:
            entries.append("<INVENTORYENTRIES.LIST>")
            entries.append(f"<STOCKITEMNAME>{escape(str(it.get('stock_item', it.get('desc', 'Item'))))}</STOCKITEMNAME>")
            entries.append(f"<QUANTITY>{it.get('qty', 0)} {escape(str(it.get('unit', 'Nos')))}</QUANTITY>")
            entries.append(f"<RATE>{_v(it.get('rate', 0))}</RATE>")
            entries.append(f"<AMOUNT>{_v(taxable)}</AMOUNT>")
            entries.append("</INVENTORYENTRIES.LIST>")

    parts.extend(entries)
    parts.append("</VOUCHER>")
    parts.append("</TALLYMESSAGE>")
    return "\n".join(parts)


def ledger_master_xml(name: str, parent: str = "Sundry Creditors", gstin: str = "") -> str:
    """TALLYMESSAGE to auto-create a ledger master (e.g. missing party or tax ledger)."""
    return "\n".join([
        '<TALLYMESSAGE xmlns:UDF="TallyUDF">',
        f'<LEDGER NAME="{escape(name)}" REMOTEID="GSTENGINE-LEDGER-{escape(name)}" ACTION="Create">',
        f"<NAME>{escape(name)}</NAME>",
        f"<PARENT>{escape(parent)}</PARENT>",
        (f"<GSTREGISTRATIONNUMBER>{escape(gstin)}</GSTREGISTRATIONNUMBER>" if gstin else ""),
        "<ISBILLWISEON>Yes</ISBILLWISEON>",
        "</LEDGER>",
        "</TALLYMESSAGE>",
    ])


def envelope(request_data: str, company: str = "") -> str:
    """Wrap TALLYMESSAGE(s) in the standard Import Data envelope."""
    sv = f"<SVCURRENTCOMPANY>{escape(company)}</SVCURRENTCOMPANY>" if company else ""
    return "\n".join([
        "<ENVELOPE>",
        "<HEADER><TALLYREQUEST>Import Data</TALLYREQUEST></HEADER>",
        "<BODY><IMPORTDATA>",
        "<REQUESTDESC>",
        "<REPORTNAME>Vouchers</REPORTNAME>",
        f"<STATICVARIABLES>{sv}</STATICVARIABLES>",
        "</REQUESTDESC>",
        f"<REQUESTDATA>{request_data}</REQUESTDATA>",
        "</IMPORTDATA></BODY>",
        "</ENVELOPE>",
    ])


def build_import(invoice: dict, voucher_type: str = "Purchase", company: str = "",
                 include_inventory: bool = False, ledger_map: dict | None = None,
                 auto_create_ledgers: bool = True) -> str:
    """
    Full import payload for one invoice: optional ledger-master creation + voucher.
    """
    lm = ledger_map or {}
    msgs = []
    if auto_create_ledgers:
        party_raw = str(invoice.get("party_name", invoice.get("gstin_supplier", "")))
        if party_raw and party_raw not in lm:
            msgs.append(ledger_master_xml(party_raw, "Sundry Creditors", invoice.get("gstin_supplier", "")))
        # Ensure standard tax and purchase ledgers exist in TallyPrime
        msgs.append(ledger_master_xml("Purchase Accounts", "Purchase Accounts"))
        msgs.append(ledger_master_xml("CGST", "Duties & Taxes"))
        msgs.append(ledger_master_xml("SGST", "Duties & Taxes"))
        msgs.append(ledger_master_xml("IGST", "Duties & Taxes"))
    msgs.append(voucher_xml(invoice, voucher_type, include_inventory, lm))
    return envelope("\n".join(msgs), company)


if __name__ == "__main__":
    demo = {
        "gstin_supplier": "27ABCTY1234E1ZL",
        "invoice_no": "ST/2026/0417",
        "date": "12-10-2026",
        "place_of_supply": "29",
        "party_name": "Sharma Traders",
        "items": [{"hsn": "8544", "desc": "Copper wire 1.5mm", "qty": 200, "unit": "Mtr",
                   "rate": 142.0, "taxable": 28400.0, "gst_rate": 18.0}],
        "total": 33512.0,
    }
    from gst_validator import validate_invoice
    report = validate_invoice(demo)
    demo["info"] = report["info"]
    xml = build_import(demo, voucher_type="Purchase", company="Demo Company",
                       include_inventory=True)
    print(xml)
