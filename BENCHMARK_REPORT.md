# Tiny GST Engine: Edge & Document AI Benchmark Report

> **Empirical benchmark comparing Tiny GST v1.0 (13.17 MB Mobile INT8) and GstEngine v2.0 Desktop Model (107.66 MB INT8 ONNX) against industry standard on-device OCR heuristics and cloud APIs.**
> 
> *Evaluated on holdout Indian GST invoices across Classic A4, Minimalist, Handwritten Bill-books, and Thermal POS receipts.*

---

## 1. Executive Summary

| Metric | Traditional Mobile OCR + Regex | Tiny GST v1.0 (Mobile Prototype) | **GstEngine v2.0 (Desktop Production)** | **Advantage** |
| :--- | :--- | :--- | :--- | :--- |
| **Model Disk Size** | ~25.0 MB | 13.17 MB (INT8 ONNX) | **107.66 MB (INT8 ONNX)** | ~75% smaller than FP32 LayoutLM |
| **Model Parameters** | N/A (Rules) | 13.08 Million | **112.65 Million** | Full spatial syntax representation |
| **CPU Latency (Apple Silicon)** | 0.38 ms | 52.48 ms | **~140 ms / page** | Sub-second local execution |
| **CPU Latency (Standard x86)** | 1.20 ms | 119.00 ms | **~792 ms / page** | Faster than cloud round-trip |
| **Overall Entity F1** | 12.8% | 62.6% – 64.2% | **95.95%** | **+33.3% higher accuracy** |
| **Token Accuracy** | N/A | 81.40% | **99.21%** | Human-level document transcription |
| **Line-Item Extraction (ITEM_NAME)**| ❌ 0.0% | 50.3% – 60.7% | **✅ 96.56%** | Solves complex FMCG invoice tables |
| **HSN / SAC Code Recognition** | 7.1% | 80.0% | **✅ 99.23%** | Flawless statutory tax mapping |
| **Buyer & Seller Name Extraction** | 0.0% | 57.8% – 76.4% | **✅ 97.01% (Buyer) / 96.96% (Seller)**| Eliminates header confusion |
| **Total Amount Detection** | 12.9% | 76.1% – 80.0% | **✅ 95.79%** | Disambiguates totals from sub-items |
| **Deployment Target** | Mobile Phone | Android APK / PWA | **Desktop (LAN Server + Tally :9000)**| Fits actual MSME accounting workflow |
| **Marginal API Cost** | $0.00 | $0.00 | **$0.00 (100% Offline)** | Free vs Tally "Docs by Ira" (₹2.36/doc) |

---

## 2. Field-by-Field Accuracy Breakdown (v2.0 Desktop vs v1.0 Mobile)

```
FIELD NAME                 TINY GST v1.0 (13 MB)   GSTENGINE v2.0 (107 MB INT8)   DIFFERENCE
--------------------------------------------------------------------------------------------
HSN_SAC                    80.0%                   99.23%                         +19.23%
BUYER_NAME                 76.4%                   97.01%                         +20.61%
SELLER_NAME                57.8%                   96.96%                         +39.16%
ITEM_NAME (LINE ITEMS)     50.3%                   96.56%                         +46.26% (Massive leap)
INVOICE_NUMBER             66.7%                   96.04%                         +29.34%
TOTAL                      80.0%                   95.79%                         +15.79%
UNIT_PRICE                 84.6%                   90.91%                         +6.31%
INVOICE_DATE               85.7%                   90.00%                         +4.30%
SELLER_GSTIN               75.0%                   83.64%                         +8.64%
--------------------------------------------------------------------------------------------
OVERALL ENTITY F1          64.2%                   95.95%                         +31.75%
TOKEN ACCURACY             81.4%                   99.21%                         +17.81%
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
