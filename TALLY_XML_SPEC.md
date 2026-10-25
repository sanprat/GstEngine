# Tally XML Import — Requirements & Field Mapping

What an XML needs to be accepted by TallyPrime's Import Data interface, and how the
GST-engine model output maps into it. Verified against TallyHelp (help.tallysolutions.com)
Oct 2026. The offline Android app writes these XML files into Tally's import folder
(file-based import) or POSTs them to Tally's HTTP endpoint (`http://<tally-host>:9000`).

## 1. Envelope (required wrapper for EVERY import)

```xml
<ENVELOPE>
  <HEADER><TALLYREQUEST>Import Data</TALLYREQUEST></HEADER>
  <BODY><IMPORTDATA>
    <REQUESTDESC>
      <REPORTNAME>Vouchers</REPORTNAME>
      <STATICVARIABLES><SVCURRENTCOMPANY>Company Name</SVCURRENTCOMPANY></STATICVARIABLES>
    </REQUESTDESC>
    <REQUESTDATA>
      ... one or more <TALLYMESSAGE> blocks ...
    </REQUESTDATA>
  </IMPORTDATA></BODY>
</ENVELOPE>
```

`SVCURRENTCOMPANY` is optional — without it Tally imports into the currently open company.

## 2. Voucher (the accounting transaction)

```xml
<TALLYMESSAGE xmlns:UDF="TallyUDF">
  <VOUCHER REMOTEID="GSTENGINE-<invoice_no>" VCHTYPE="Purchase" ACTION="Create" OBJVIEW="Accounting Voucher View">
    <DATE>20261020</DATE>            <!-- YYYYMMDD -->
    <EFFECTIVEDATE>20261020</EFFECTIVEDATE>
    <VOUCHERTYPENAME>Purchase</VOUCHERTYPENAME>
    <VOUCHERNUMBER>261027D011121904</VOUCHERNUMBER>
    <NARRATION>GST Engine auto-import ...</NARRATION>
    <PERSISTEDVIEW>Accounting Voucher View</PERSISTEDVIEW>
    <!-- one entry per ledger line -->
    <ALLLEDGERENTRIES.LIST>
      <LEDGERNAME>Drogheria Sellers Private Limited</LEDGERNAME>
      <ISDEEMEDPOSITIVE>Yes</ISDEEMEDPOSITIVE>   <!-- Yes = credit side (party) -->
      <AMOUNT>-103.00</AMOUNT>
    </ALLLEDGERENTRIES.LIST>
    <ALLLEDGERENTRIES.LIST>
      <LEDGERNAME>Purchase Accounts</LEDGERNAME>
      <ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE>    <!-- No = debit side -->
      <AMOUNT>98.10</AMOUNT>
    </ALLLEDGERENTRIES.LIST>
    <ALLLEDGERENTRIES.LIST>
      <LEDGERNAME>CGST</LEDGERNAME><ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE><AMOUNT>2.45</AMOUNT>
    </ALLLEDGERENTRIES.LIST>
    <ALLLEDGERENTRIES.LIST>
      <LEDGERNAME>SGST</LEDGERNAME><ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE><AMOUNT>2.45</AMOUNT>
    </ALLLEDGERENTRIES.LIST>
  </VOUCHER>
</TALLYMESSAGE>
```

**Required semantics:**
- `REMOTEID` — **idempotency key**. Re-importing the same REMOTEID → Tally IGNOREs it (no double entry). Always derive from invoice number.
- `ISDEEMEDPOSITIVE` — decides debit/credit side. **Party = Yes (credit, negative amount)**, expense/tax ledgers = No (debit, positive).
- Debits must equal credits (Tally enforces; a ₹0.01 rounding diff can fail import — see §5).
- Intra-state: CGST + SGST (each half). Inter-state: IGST (single).
- Optionally `<INVENTORYENTRIES.LIST>` per item with STOCKITEMNAME/QUANTITY/RATE/AMOUNT for stock-in tracking.

## 3. Ledger master (auto-create when a party/tax ledger doesn't exist)

```xml
<TALLYMESSAGE xmlns:UDF="TallyUDF">
  <LEDGER NAME="Drogheria Sellers Private Limited" REMOTEID="GSTENGINE-LEDGER-..." ACTION="Create">
    <NAME>Drogheria Sellers Private Limited</NAME>
    <PARENT>Sundry Creditors</PARENT>
    <GSTREGISTRATIONNUMBER>27AAJCD2242F1ZW</GSTREGISTRATIONNUMBER>
    <ISBILLWISEON>Yes</ISBILLWISEON>
  </LEDGER>
</TALLYMESSAGE>
```
Send this BEFORE the voucher if the party is new (Tally errors "Ledger does not exist" otherwise).

## 4. Model field → XML mapping

| Model label (BIOES) | Invoice dict key | XML element | Notes |
|---|---|---|---|
| SELLER_NAME | party_name | LEDGER NAME / LEDGERNAME | The supplier we purchased from |
| SELLER_GSTIN | gstin_supplier | LEDGER GSTREGISTRATIONNUMBER | Also used for inter-state check |
| INVOICE_NUMBER | invoice_no | VOUCHERNUMBER + REMOTEID | Idempotency key |
| INVOICE_DATE | date | DATE / EFFECTIVEDATE | converted to YYYYMMDD |
| PLACE_OF_SUPPLY | place_of_supply | (derived) | state name → 2-digit code |
| ITEM_NAME | items[].desc | INVENTORYENTRIES STOCKITEMNAME / narration | |
| HSN_SAC | items[].hsn | (inventory; optional in ledger view) | |
| QUANTITY | items[].qty | INVENTORYENTRIES QUANTITY | |
| UNIT_PRICE | items[].rate | INVENTORYENTRIES RATE | |
| DISCOUNT | items[].discount | (adjusts taxable; not a ledger line) | |
| TAXABLE_VALUE | items[].taxable | Purchase Accounts AMOUNT | the debit |
| CGST_AMOUNT / SGST_AMOUNT | items[].cgst_amt / sgst_amt | CGST / SGST ledger AMOUNT | verbatim from invoice |
| IGST_AMOUNT | items[].igst_amt | IGST ledger AMOUNT | inter-state only |
| TOTAL | total | party AMOUNT (negative) | credit side |
| CREDIT/DEBIT_NOTE_NUMBER | (future) | VCHTYPE Credit Note / Debit Note | |

**Inter/intra-state rule:** supplier GSTIN first 2 chars vs place-of-supply code. Different → IGST; same → CGST+SGST.

## 5. Rounding pitfall (verified on real Zepto invoice)

The invoice's own per-line rounding (CGST 2.45 + SGST 2.45 = 4.90 on taxable 98.10) differs
from rate-computed (98.10 × 5% = 4.905 → CGST 2.46 + SGST 2.45). **Always pass the extracted
CGST/SGST/IGST amounts verbatim** (`cgst_amt`/`sgst_amt`/`igst_amt`) instead of recomputing
from `gst_rate`; fall back to rate-compute only when amounts are missing. Tally requires
debits == credits exactly.

## 6. Integration paths

1. **File-based (offline Android):** write the envelope to an `.xml` file; user drops it into
   Tally's import folder (Alt+O → Import Data → Vouchers). No network, matches the app's offline constraint.
2. **HTTP (Tally running on same LAN):** POST the envelope to `http://<host>:9000`.
   Response `<IMPORTRESULT>` contains `<CREATED>`, `<ALTERED>`, `<IGNORED>`, or `<LINEERROR>`.
   On "Ledger does not exist" → send ledger-master TALLYMESSAGE, then retry the voucher once.

## Implementation

- `tally_xml.py` — envelope + voucher + ledger-master builders (done, verified against TallyHelp).
- `tally_client.py` — POST + parse CREATED/ALTERED/IGNORED/LINEERROR + missing-ledger recovery (done).
- `mock_tally.py` — local test server simulating Tally's :9000 (done; used for the demo).
- `extract_to_tally.py` — **the bridge**: BIOES-labeled record → validated invoice dict → XML.
  Run: `python extract_to_tally.py <image_id> --company "X" --out /tmp/imp.xml [--post 127.0.0.1:9000]`
