import json, os
from collections import Counter
import test_label_ocr

dataset_path = 'data/gst_smoke_dataset.jsonl'
stats_path = 'data/gst_smoke_stats.json'

with open(dataset_path) as f:
    existing_recs = [json.loads(line) for line in f if line.strip()]

print(f"Existing records: {len(existing_recs)}")
existing_ids = {r['image_id'] for r in existing_recs}

new_recs = [r for r in test_label_ocr.labeled_records if r['image_id'] not in existing_ids]
print(f"Adding {len(new_recs)} new records from image OCR...")

merged_recs = existing_recs + new_recs

# Write merged dataset
with open(dataset_path, 'w') as f:
    for r in merged_recs:
        f.write(json.dumps(r) + '\n')

# Compute updated stats
vendor_counts = Counter(r['vendor'] for r in merged_recs)
label_counts = Counter(l.split('-')[-1] for r in merged_recs for l in r['labels'])

stats = {
    "pages": len(merged_recs),
    "vendors": dict(vendor_counts),
    "label_counts": dict(label_counts),
}

with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"Merged dataset saved: {len(merged_recs)} total pages across {len(vendor_counts)} distinct vendors.")
print("Distinct vendors:", len(vendor_counts))
