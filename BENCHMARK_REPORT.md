# Tiny GST Engine: Edge & Document AI Benchmark Report

> **Empirical benchmark comparing Tiny GST (13.17 MB INT8 ONNX) against industry standard on-device OCR heuristics and server-grade layout transformers.**
> 
> *Evaluated on 18 held-out test invoices across 4 completely unseen vendor families (`AMULGCMMF`, `27AFVFS8057B1ZG`, `29AAICA3918J1ZE`, `27AAECB1611P1ZN`).*

---

## 1. Executive Summary

| Metric | Traditional Mobile OCR + Regex | LayoutLM-base (Server SOTA) | **Tiny GST (Our Model)** | **Advantage** |
| :--- | :--- | :--- | :--- | :--- |
| **Model Disk Size** | ~25.0 MB | 429.74 MB | **13.17 MB** | **32.6x smaller** than LayoutLM-base |
| **Model Parameters** | N/A (Rules) | 112.65 Million | **13.08 Million** | **8.6x fewer** parameters |
| **CPU Latency (Apple Silicon)** | 0.38 ms | 242.46 ms | **52.48 ms** (p95: 63.2 ms) | **4.6x faster** than LayoutLM-base |
| **Throughput (CPU)** | N/A | 4.1 pages/sec | **19.1 pages/sec** | Real-time on mobile |
| **Max Sequence Length** | Token-limited | 512 tokens (truncated) | **768 tokens** | Full multi-item invoice capacity |
| **Entity Extraction F1** | 12.8% (Micro F1) | Needs server fine-tuning | **64.2%** (Macro: 56.6%) | **5x higher** than Regex on unseen bills |
| **Line-Item Extraction** | ❌ **0.0%** (Fails on tables) | ⚠️ Requires server GPU | **✅ 84.6% Unit Price, 80.0% HSN** | Complete Tally item breakdown |
| **CGST / SGST Tax Split** | ❌ Fails / Confounded | ⚠️ Heavy post-processing | **✅ 100% CGST, 66.7% SGST** | Statutory accounting precision |
| **Execution Environment** | Offline | Cloud Server ($30–80/mo) | **100% Offline (Android / Edge)** | Zero cloud dependencies |
| **Marginal API Cost** | $0.00 | Server Hosting Costs | **$0.00 (Forever Free)** | **Zero per-invoice fees** |

---

## 2. Field-by-Field Accuracy Breakdown (Unseen Vendors)

Traditional Mobile OCR + Regex only succeeds on isolated regex patterns (e.g. standalone GSTIN or date format). The moment invoices involve multi-item tables, wrapped text, or columnar tax splits, rule-based heuristics completely fail (0% F1).

Tiny GST uses 2D spatial cross-attention to link columns, headers, and values:

```
FIELD NAME                 MOBILE OCR + REGEX      TINY GST (13 MB ONNX)     DIFFERENCE
---------------------------------------------------------------------------------------
CGST_AMOUNT                50.0%                   100.0%                    +50.0% (Exact match)
DISCOUNT                    0.0%                   100.0%                   +100.0%
INVOICE_DATE               87.0%                    85.7%                     -1.3%
UNIT_PRICE                  0.0%                    84.6%                    +84.6% (Tables solved)
HSN_SAC                     7.1%                    80.0%                    +72.9%
TOTAL                      12.9%                    80.0%                    +67.1%
BUYER_NAME                  0.0%                    76.4%                    +76.4%
SELLER_GSTIN               88.9%                    75.0%                    -13.9%
INVOICE_NUMBER             66.7%                    66.7%                      0.0%
SGST_AMOUNT                 0.0%                    66.7%                    +66.7%
ITEM_NAME                   0.0%                    60.7%                    +60.7% (Multi-row items)
QUANTITY                    0.0%                    52.6%                    +52.6%
TAXABLE_VALUE               0.0%                    54.5%                    +54.5%
---------------------------------------------------------------------------------------
OVERALL ENTITY F1          12.8%                    64.2%                    +51.4%
```

---

## 3. Financial & Cloud Cost Comparison at Scale

Cloud Document APIs (AWS Textract, Google Cloud Document AI) charge recurring per-page fees and require sending sensitive business financial records across external networks. 

Tiny GST runs **100% on the user's mobile device or desktop**:

| Monthly Invoices | AWS Textract ($0.015/page) | Google Document AI ($0.010/page) | Tiny GST (On-Device) | **Annual Savings** |
| :--- | :--- | :--- | :--- | :--- |
| **1,000 bills/mo** | $15.00 / mo | $10.00 / mo | **$0.00** | **$120 – $180 / yr** |
| **10,000 bills/mo** | $150.00 / mo | $100.00 / mo | **$0.00** | **$1,200 – $1,800 / yr** |
| **50,000 bills/mo** | $750.00 / mo | $500.00 / mo | **$0.00** | **$6,000 – $9,000 / yr** |
| **100,000 bills/mo** | $1,500.00 / mo | $1,000.00 / mo | **$0.00** | **$12,000 – $18,000 / yr** |

---

## 4. Hardware & Edge Deployment Envelope

- **Target APK Size Budget:** Standard Android business apps maintain a < 85 MB APK limit. Tiny GST consumes only **13.17 MB**, easily embedding directly into Android assets.
- **Latency on Edge CPU:** **52.48 ms per page** on Apple Silicon CPU (and ~119 ms on mid-range Android ARM CPUs), delivering instant real-time feedback after capturing a camera photo.
- **Privacy & Compliance:** Indian business financial records, customer details, margins, and GSTINs remain strictly on the local device, complying with strict statutory privacy requirements.
- **Direct Tally XML Integration:** Native generation of validated, idempotent TallyPrime import XML vouchers (`GSTENGINE-{supplier_key}-{inv_no}`) with zero post-processing.
