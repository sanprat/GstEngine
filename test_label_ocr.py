import json, re
from collections import Counter

with open('invoices/ocr_batch_8.json') as f:
    ocr_data = json.load(f)

# Ground truth metadata for each image
GT_META = {
    "IMG_20261024_232850375.jpg": {
        "vendor": "27AAECB1611P1ZN",
        "doc_type": "invoice",
        "invoice_number": "FASZO22602287013",
        "seller_name_kw": ["Buzz", "Retall", "India", "Private", "Limited"],
        "seller_gstin_raw": "27AAECB161PIZN",
        "invoice_num_raw": "FASZO22602287013",
        "date_raw": "03-01-2026",
        "buyer_raw": ["TestBuyer", "TestBuyer", "Buyer"],
        "items": [
            {"desc_kw": ["FUNSKOOL", "Othello"], "hsn": "95030020", "total": "301.00"}
        ],
        "grand_total": "301.00"
    },
    "IMG_20261024_232933832.jpg": {
        "vendor": "06AFDPJ6395R1ZO",
        "doc_type": "invoice",
        "invoice_number": "OD432377970165767100",
        "seller_name_kw": ["Verdhman", "Wool", "Store"],
        "seller_gstin_raw": "O6AFDPJ6395r1z0",
        "invoice_num_raw": "OD432377970165767100",
        "date_raw": "21-09-2024",
        "buyer_raw": ["CustomerTwo"],
        "items": [
            {"desc_kw": ["Kuchipoo", "Girls"], "hsn": "61099090", "total": "579.00"}
        ],
        "grand_total": "579.00"
    },
    "IMG_20261024_233028767.jpg": {
        "vendor": "27AAKCC0172C2Z0",
        "doc_type": "invoice",
        "invoice_number": "FaNO012500052728",
        "seller_name_kw": ["CLOUDSTORE", "RETAIL", "PRivATE", "LIMITED"],
        "seller_gstin_raw": "27AAKCCQ172C2Z0",
        "invoice_num_raw": "FaNO012500052728",
        "date_raw": "19-07-2024",
        "buyer_raw": ["TESTBUYER", "BUYER"],
        "items": [
            {"desc_kw": ["Nothing", "Phone", "(2a)"], "hsn": "85171300", "total": "19349.00"}
        ],
        "grand_total": "19349.00"
    },
    "IMG_20261024_233040515.jpg": {
        "vendor": "29AABCW7091B2Z7",
        "doc_type": "invoice",
        "invoice_number": "FAIHJX2400007404",
        "seller_name_kw": ["Wishery", "Online", "Private", "Limited"],
        "seller_gstin_raw": "Z9AABCW709182Z7",
        "invoice_num_raw": "FAIHJX2400007404",
        "date_raw": "18-04-2023",
        "buyer_raw": ["TESTBUYER", "BUYER"],
        "items": [
            {"desc_kw": ["Poco", "M4"], "hsn": "85171300", "total": "9164.00"}
        ],
        "grand_total": "9164.00"
    },
    "IMG_20261024_233054621.jpg": {
        "vendor": "29AATCA0996P1ZJ",
        "doc_type": "invoice",
        "invoice_number": "FAIMGD740O144591",
        "seller_name_kw": ["AARNA", "VENTURES", "PRIVATE", "LIMITED"],
        "seller_gstin_raw": "ZOAATCAQ9O6PIZJ",
        "invoice_num_raw": "FAIMGD740O144591",
        "date_raw": "21-03-2024",
        "buyer_raw": ["Muneera", "Banu"],
        "items": [
            {"desc_kw": ["REDMI", "A2"], "hsn": "85171300", "total": "5299.00"}
        ],
        "grand_total": "5299.00"
    },
    "IMG_20261024_233104778.jpg": {
        "vendor": "27AAICB4842N1Z8",
        "doc_type": "invoice",
        "invoice_number": "FAJH742501670936",
        "seller_name_kw": ["Bhumika", "Hlghstreet", "Indla", "Private", "Limited"],
        "seller_gstin_raw": "Z7AAICB484ZN1Z8",
        "invoice_num_raw": "FAJH742501670936",
        "date_raw": "03-02-2025",
        "buyer_raw": ["TestBuyer", "TestBuyer", "Buyer"],
        "items": [
            {"desc_kw": ["Formal", "Shoes"], "hsn": "64039990", "total": "1669.00"}
        ],
        "grand_total": "1669.00"
    },
    "IMG_20261024_233112896.jpg": {
        "vendor": "27AAXCS8712M1ZD",
        "doc_type": "invoice",
        "invoice_number": "FAFP8H260o069592",
        "seller_name_kw": ["Savan", "Retailers", "Private", "Limited"],
        "seller_gstin_raw": "27AAXCS87t2izd",
        "invoice_num_raw": "FAFP8H260o069592",
        "date_raw": "23-11-2025",
        "buyer_raw": ["TestBuyer", "TestBuyer", "Buyer"],
        "items": [
            {"desc_kw": ["vivo", "Tax", "56"], "hsn": "85171300", "total": "12752.00"}
        ],
        "grand_total": "12752.00"
    },
    "IMG_20261024_233128899.jpg": {
        "vendor": "07BOXPJ1735H1ZL",
        "doc_type": "invoice",
        "invoice_number": "F34-01200003247",
        "seller_name_kw": ["NOBARR", "INDIA"],
        "seller_gstin_raw": "OTROXPIITJSHIZL",
        "invoice_num_raw": "F34-01200003247",
        "date_raw": "24",
        "buyer_raw": ["SANII", "HUSSAN", "SANil}", "Hassah"],
        "items": [
            {"desc_kw": ["Noberq", "Women", "Checked"], "hsn": "Hn27032090", "total": "389"}
        ],
        "grand_total": "389"
    }
}

def set_span(labels, i, j, field):
    if j < i: return
    if j == i:
        labels[i] = f"S-{field}"
    else:
        labels[i] = f"B-{field}"
        for k in range(i+1, j):
            labels[k] = f"I-{field}"
        labels[j] = f"E-{field}"

labeled_records = []

for img_name, meta in sorted(GT_META.items()):
    data = ocr_data[img_name]
    words = data['words']
    boxes = data['boxes']
    n = len(words)
    labels = ["O"] * n
    used = set()

    # 1. Seller Name
    seller_kw = meta["seller_name_kw"]
    # find where seller_kw sequence occurs
    for i in range(len(words) - len(seller_kw) + 1):
        if all(words[i+k].strip(',.') == seller_kw[k] for k in range(len(seller_kw))):
            set_span(labels, i, i + len(seller_kw) - 1, "SELLER_NAME")
            used.update(range(i, i + len(seller_kw)))
            break

    # 2. Seller GSTIN
    gstin_raw = meta["seller_gstin_raw"]
    for i, w in enumerate(words):
        if w.strip(',.') == gstin_raw:
            set_span(labels, i, i, "SELLER_GSTIN")
            used.add(i)
            break

    # 3. Invoice Number
    inv_raw = meta["invoice_num_raw"]
    for i, w in enumerate(words):
        if inv_raw.lower() in w.lower():
            set_span(labels, i, i, "INVOICE_NUMBER")
            used.add(i)
            break

    # 4. Invoice Date
    date_raw = meta["date_raw"]
    for i, w in enumerate(words):
        if date_raw in w and i not in used:
            set_span(labels, i, i, "INVOICE_DATE")
            used.add(i)
            break

    # 5. Buyer Name
    buyer_kw = meta["buyer_raw"]
    for i in range(len(words)):
        if words[i] in buyer_kw and i not in used:
            # check if adjacent buyer token
            if i + 1 < len(words) and words[i+1] in buyer_kw and (i+1) not in used:
                set_span(labels, i, i+1, "BUYER_NAME")
                used.add(i); used.add(i+1)
            else:
                set_span(labels, i, i, "BUYER_NAME")
                used.add(i)

    # 6. HSN
    for item in meta["items"]:
        hsn = item["hsn"]
        for i, w in enumerate(words):
            if hsn in w and i not in used:
                set_span(labels, i, i, "HSN_SAC")
                used.add(i)
                break

    # 7. Item Name
    for item in meta["items"]:
        desc_kw = item["desc_kw"]
        for i in range(len(words) - len(desc_kw) + 1):
            if all(words[i+k].strip(',.') == desc_kw[k] for k in range(len(desc_kw))) and not any(i+k in used for k in range(len(desc_kw))):
                set_span(labels, i, i + len(desc_kw) - 1, "ITEM_NAME")
                used.update(range(i, i + len(desc_kw)))
                break

    # 8. Grand Total
    gt = meta["grand_total"]
    # find where gt appears after word index 100
    for i in range(len(words)-1, -1, -1):
        if words[i].strip(',.') == gt and i not in used:
            set_span(labels, i, i, "TOTAL")
            used.add(i)
            break

    rec = {
        "image_id": img_name.replace(".jpg", "_p1"),
        "vendor": meta["vendor"],
        "doc_type": meta["doc_type"],
        "invoice_number": meta["invoice_number"],
        "words": words,
        "boxes": boxes,
        "labels": labels
    }
    labeled_records.append(rec)

print(f"Successfully labeled {len(labeled_records)} records!")
for r in labeled_records:
    labeled_fields = [l for l in r['labels'] if l != 'O']
    print(f"{r['image_id']} (vendor {r['vendor']}): {len(labeled_fields)} non-O labels -> {Counter(labeled_fields)}")
