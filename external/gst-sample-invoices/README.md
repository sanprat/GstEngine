# Indian GST Invoices Benchmark & Testing Dataset

A comprehensive, dual-source collection of Indian GST (Goods and Services Tax) invoices designed specifically for testing, benchmarking, and evaluating OCR (Tesseract, PaddleOCR), Document AI (LayoutLM, Donut), and Multimodal LLMs (GPT-4o, Claude 3.5 Sonnet, Gemini Flash).

---

## 📁 Directory Structure & Download

All datasets are saved in:  
**`gst-sample-invoices/gst-sample-invoices/`**

📦 **All-In-One ZIP Archive**: [`gst_sample_invoices.zip`](file://gst-sample-invoices/gst-sample-invoices/gst_sample_invoices.zip) *(Contains all 27+ invoices across synthetic and real internet datasets)*

```text
gst-sample-invoices/
├── GST_Invoice_01_TechHardware_B2B_Intrastate.[pdf|png|html]
├── GST_Invoice_02_Industrial_Interstate_eInvoice.[pdf|png|html]
├── GST_Invoice_03_Consulting_SaaS_Services.[pdf|png|html]
├── GST_Invoice_04_Pharma_MultiRate_Batches.[pdf|png|html]
├── ground_truth_dataset.json
├── evaluate_model.py
├── internet_samples/
│   ├── internet_gst_invoice_01.jpg ... internet_gst_invoice_22.jpg
│   ├── github_sample_gst_invoice.png
│   └── internet_samples_ground_truth.json
└── gst_sample_invoices.zip
```

---

## 🌐 1. Real-World Internet Dataset (`internet_samples/`)

Extracted directly from open benchmark repositories (HuggingFace `AjitRawat/invoice` and open-source GitHub billing projects):

- **Count**: **22 Real-World Indian Invoices** (`.jpg`) + **1 GitHub Standard Invoice** (`.png`).
- **Real-World Variations**: Scanned documents, digital receipts, skewed layouts, variable resolutions, real vendor tax invoices (Dell India, hardware vendors, service bills).
- **Ground Truth**: [`internet_samples/internet_samples_ground_truth.json`](file://gst-sample-invoices/gst-sample-invoices/internet_samples/internet_samples_ground_truth.json)  
  Includes annotations for:
  - `invoice_date`
  - `gstin_no`
  - `pan`
  - `amount`
  - `address`
  - `order_no` / `customer_no`
  - Line-by-line `item_desc`

---

## 📑 2. Standardized High-Fidelity Test Suite

Clean, mathematically exact vector PDFs, PNGs, and responsive HTMLs with complete e-Invoice compliance:

1. **Tech Hardware (B2B Intra-State)**: [`GST_Invoice_01_TechHardware_B2B_Intrastate.pdf`](file://gst-sample-invoices/gst-sample-invoices/GST_Invoice_01_TechHardware_B2B_Intrastate.pdf)  
   - Intra-state **CGST (9%) + SGST (9%)**, HSN codes (`8471`, `8528`), line discounts, bank IFSC/NEFT details.
2. **Industrial Equipment (B2B Inter-State e-Invoice)**: [`GST_Invoice_02_Industrial_Interstate_eInvoice.pdf`](file://gst-sample-invoices/gst-sample-invoices/GST_Invoice_02_Industrial_Interstate_eInvoice.pdf)  
   - Inter-state **IGST (18%)**, authentic 64-char IRN hash, Ack No & Date, E-Way Bill barcode, e-Invoice QR code.
3. **IT Consulting & SaaS**: [`GST_Invoice_03_Consulting_SaaS_Services.pdf`](file://gst-sample-invoices/gst-sample-invoices/GST_Invoice_03_Consulting_SaaS_Services.pdf)  
   - Services **SAC codes** (`998313`, `998314`), monthly retainers, service periods.
4. **Healthcare & Pharmaceuticals**: [`GST_Invoice_04_Pharma_MultiRate_Batches.pdf`](file://gst-sample-invoices/gst-sample-invoices/GST_Invoice_04_Pharma_MultiRate_Batches.pdf)  
   - Multi-tier GST slabs (**5%, 12%, 18%**), Batch Numbers, Expiry Dates, MRP, Drug License numbers.

---

## ⚡ Automated Benchmark Evaluation

To check your model's accuracy:

```bash
cd gst-sample-invoices/gst-sample-invoices
python3 evaluate_model.py
```
