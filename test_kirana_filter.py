import json, re, sys, os
from collections import Counter
import numpy as np

def process_kirana(max_samples=100, min_labeled_tokens=8):
    from datasets import load_dataset
    import easyocr

    print(f"Loading kirana-invoice-train-data (evaluating {max_samples} samples)...")
    ds = load_dataset('build-small-hackathon/kirana-invoice-train-data')
    train_ds = ds['train']

    reader = easyocr.Reader(['en'], gpu=True)
    clean_records = []
    skipped = 0

    for idx in range(min(max_samples, len(train_ds))):
        sample = train_ds[idx]
        img = sample['image']
        w_img, h_img = img.size
        resp = json.loads(sample['response'])

        img_np = np.array(img)
        ocr_results = reader.readtext(img_np)

        words, boxes = [], []
        for bbox, text, prob in ocr_results:
            text = text.strip()
            if not text:
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            toks = text.split()
            if len(toks) <= 1:
                bx0 = max(0, min(1000, int(round(min_x * 1000.0 / w_img))))
                by0 = max(0, min(1000, int(round(min_y * 1000.0 / h_img))))
                bx1 = max(bx0 + 1, min(1000, int(round(max_x * 1000.0 / w_img))))
                by1 = max(by0 + 1, min(1000, int(round(max_y * 1000.0 / h_img))))
                words.append(text); boxes.append([bx0, by0, bx1, by1])
            else:
                total_chars = sum(len(w) for w in toks)
                cur_x = min_x
                box_w = max_x - min_x
                for w in toks:
                    w_len = len(w)
                    char_ratio = w_len / max(1, total_chars)
                    word_w = box_w * char_ratio
                    bx0 = max(0, min(1000, int(round(cur_x * 1000.0 / w_img))))
                    by0 = max(0, min(1000, int(round(min_y * 1000.0 / h_img))))
                    bx1 = max(bx0 + 1, min(1000, int(round((cur_x + word_w) * 1000.0 / w_img))))
                    by1 = max(by0 + 1, min(1000, int(round(max_y * 1000.0 / h_img))))
                    words.append(w); boxes.append([bx0, by0, bx1, by1])
                    cur_x += word_w

        n = len(words)
        labels = ["O"] * n
        used = set()

        def set_span(i, j, field):
            if j < i or any(k in used for k in range(i, j+1)): return
            if j == i:
                labels[i] = f"S-{field}"
            else:
                labels[i] = f"B-{field}"
                for k in range(i+1, j):
                    labels[k] = f"I-{field}"
                labels[j] = f"E-{field}"
            used.update(range(i, j+1))

        # 1. Supplier / Seller Name
        supplier = resp.get("supplier") or ""
        supp_words = supplier.split()
        if supp_words:
            for i in range(len(words) - len(supp_words) + 1):
                if all(words[i+k].strip(',.').lower() == supp_words[k].strip(',.').lower() for k in range(len(supp_words))):
                    set_span(i, i + len(supp_words) - 1, "SELLER_NAME")
                    break

        # 2. Buyer Name (look for "To: [Buyer Name]" in words or match retailer)
        for i in range(len(words) - 2):
            if words[i].lower() in ("to:", "to", "bill to:") and i not in used:
                end_k = min(i + 4, len(words))
                set_span(i + 1, end_k - 1, "BUYER_NAME")
                break

        # 3. Invoice Number
        inv_no = resp.get("invoice_number") or ""
        if inv_no:
            for i, w in enumerate(words):
                if (inv_no.lower() in w.lower() or w.lower() in inv_no.lower()) and len(w) >= 6:
                    set_span(i, i, "INVOICE_NUMBER")
                    break

        # 4. Date
        date_str = resp.get("date") or ""
        if date_str:
            for i, w in enumerate(words):
                if date_str in w or (len(w) >= 8 and (w in date_str or date_str in w)):
                    set_span(i, i, "INVOICE_DATE")
                    break

        # 5. Items
        for item in resp.get("items", []):
            prod = item.get("product_raw") or ""
            p_words = prod.split()
            if p_words:
                for i in range(len(words) - len(p_words) + 1):
                    if all(words[i+k].strip(',.').lower() == p_words[k].strip(',.').lower() for k in range(len(p_words))):
                        set_span(i, i + len(p_words) - 1, "ITEM_NAME")
                        break
            
            qty = str(item.get("quantity"))
            rate = str(int(round(float(item.get("unit_price", 0)))))
            for i, w in enumerate(words):
                if w.strip(',.') == qty and i not in used:
                    set_span(i, i, "QUANTITY")
                    break
            for i, w in enumerate(words):
                if (w.strip(',.') == rate or w.strip(',.') == str(item.get("unit_price"))) and i not in used:
                    set_span(i, i, "UNIT_PRICE")
                    break

        # 6. Grand Total
        gt = resp.get("grand_total")
        if gt:
            gt_int = str(int(round(float(gt))))
            for i in range(len(words)-1, -1, -1):
                w_clean = words[i].strip(',.₹')
                if (w_clean == gt_int or w_clean == str(gt)) and i not in used:
                    set_span(i, i, "TOTAL")
                    break

        non_o = sum(1 for l in labels if l != "O")
        if non_o < min_labeled_tokens:
            skipped += 1
            continue

        vendor_id = re.sub(r'[^A-Z0-9]', '', (supplier or f"KIRANA_{idx}").upper())[:15]
        if len(vendor_id) < 5: vendor_id = f"KIRANA_{idx:03d}"

        rec = {
            "image_id": f"kirana_{idx:04d}_p1",
            "vendor": vendor_id,
            "doc_type": "invoice",
            "invoice_number": inv_no or f"KIRANA-{idx}",
            "words": words,
            "boxes": boxes,
            "labels": labels
        }
        clean_records.append(rec)

    print(f"\nDone! Accepted {len(clean_records)} high-quality records, skipped {skipped} noisy records.")
    return clean_records

if __name__ == "__main__":
    records = process_kirana(max_samples=100, min_labeled_tokens=8)
    print(f"Extracted {len(records)} clean records.")
    vendors = Counter(r['vendor'] for r in records)
    print(f"Vendors: {len(vendors)} distinct -> {dict(vendors)}")
