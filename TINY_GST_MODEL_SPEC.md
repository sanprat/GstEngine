# LayoutLMv1-Tiny-GST: build specification

Status: architecture and training specification. No model has been trained yet.

## Product constraints

- Commercial Android app
- Offline inference
- Existing OCR: Google ML Kit, preferably unbundled through Play Services
- One shipped extractor model
- Target installed size: 60–85 MB
- LayoutLMv1 licensing retained; do not ship LayoutLMv3 or ModernBERT

## Student model

LayoutLMv1-compatible token + 2D-layout encoder:

- Transformer layers: 6
- Hidden size: 256
- Attention heads: 4
- Intermediate size: 768
- Max sequence length: 512 initially
- Token classification head with BIOES labels
- Optional row/group head trained separately and folded into post-processing
- INT8 quantization-aware training, then ONNX Runtime Mobile export

Expected model size is a target, not a measured result. Measure the exported artifact before making a size claim.

## Runtime pipeline

ML Kit OCR -> words and bounding boxes -> normalize coordinates to 0..1000 -> tokenize while retaining word boxes -> student token classifier -> row grouping -> GST validation/reconciliation -> JSON.

The student must not receive raw image patches. This preserves the small layout-aware architecture and keeps preprocessing light.

## Labels

O, B/I/E/S for:

SELLER_NAME, SELLER_GSTIN, BUYER_NAME, BUYER_GSTIN, INVOICE_NUMBER, INVOICE_DATE,
PLACE_OF_SUPPLY, ITEM_NAME, HSN_SAC, QUANTITY, UNIT_PRICE, DISCOUNT, TAXABLE_VALUE,
CGST_RATE, CGST_AMOUNT, SGST_RATE, SGST_AMOUNT, IGST_RATE, IGST_AMOUNT, TOTAL,
CREDIT_NOTE_NUMBER, DEBIT_NOTE_NUMBER.

Keep a separate document-level classification label for invoice, credit note, debit note, receipt, hotel bill, restaurant bill, pharmacy bill, and unknown.

## Training recipe

1. Start from the existing LayoutLMv1 teacher and copy compatible embeddings/encoder weights into the 6-layer/256-width student where dimensions match; otherwise use a teacher only for soft targets.
2. Fine-tune supervised token classification on real OCR tokens and boxes.
3. Distill teacher logits with temperature 2–4.
4. Add intermediate hidden-state and attention-map distillation only where dimensions can be projected safely.
5. Add document-type classification and row-boundary auxiliary loss.
6. Use class-balanced loss or focal loss for rare GST fields.
7. Apply QAT before export; compare against dynamic INT8 quantization.
8. Export ONNX with dynamic batch=1 and fixed maximum sequence length.
9. Build a reduced-op ONNX Runtime Mobile package after the graph is frozen.

## Data plan

The supplied PDFs are valuable layout/OCR failure cases, but six invoices are not enough to train a general model. Use them for a smoke test and error taxonomy, not as the final training set.

Minimum next milestone: 100–150 labeled invoices across at least 20 vendor/layout families. Hold out entire vendors for evaluation. Include e-commerce, hotel, restaurant, pharmacy, hospital, transport, utility, retail, credit-note, debit-note, multi-page, and handwritten/low-quality cases.

Generate synthetic invoices only as augmentation. Synthetic data must vary typography, column order, missing fields, tax type, discounts, page breaks, line count, and OCR corruption. Never use synthetic-only evaluation.

## Post-processing

The rules are model-independent and should remain small:

- GSTIN regex + checksum + state-code validation
- date normalization with ambiguity flagging
- HSN/SAC shape validation, not blind dictionary rejection
- row grouping from box centers and x-order
- CGST+SGST vs IGST exclusivity checks
- subtotal/tax/total reconciliation with tolerance
- duplicate-page and marketplace-fee detection
- confidence thresholding and human-review flag

Rules must not invent values. If arithmetic conflicts with the source, preserve the extracted values and emit a discrepancy.

## Evaluation gates

Do not proceed to Android packaging until all are measured on a vendor-held-out set:

- field exact match and normalized match
- line-item precision/recall/F1
- GSTIN exact match and checksum accuracy
- total reconciliation accuracy
- document-type accuracy
- abstention/low-confidence precision
- p50/p95 latency on representative Android hardware
- exported model byte size and peak RAM

Acceptance target for first useful model: >=90% normalized header-field accuracy, >=85% line-item F1, and >=95% arithmetic validation accuracy on held-out layouts. These are gates to measure, not promises.

## Alternatives to benchmark, not ship initially

- PaddleOCR mobile structure/KIE components: Apache-licensed and potentially very small, but a multi-model pipeline; benchmark as a separate baseline.
- LiLT: layout-aware and useful for multilingual experiments, but verify checkpoint/license and deployment size before considering it.
- MobileBERT/ModernBERT: teacher-only candidates; text-only models do not replace the 2D layout input without an added layout mechanism.
- SmolVLM/Donut/VLMs: research/server fallback only; not compatible with the initial <100 MB single-model target.
