import json, os
from collections import Counter

dataset_path = 'data/gst_smoke_dataset.jsonl'
stats_path = 'data/gst_smoke_stats.json'

with open(dataset_path) as f:
    existing_recs = [json.loads(line) for line in f if line.strip()]

print(f"Existing records: {len(existing_recs)}")
existing_ids = {r['image_id'] for r in existing_recs}

kirana_part1 = [json.loads(l) for l in open('data/kirana_clean_records.jsonl') if l.strip()]
kirana_part2 = [json.loads(l) for l in open('data/kirana_clean_records_part2.jsonl') if l.strip()]
all_kirana = kirana_part1 + kirana_part2

new_recs = [r for r in all_kirana if r['image_id'] not in existing_ids]
print(f"Adding {len(new_recs)} new records from Kirana FMCG dataset...")

merged_recs = existing_recs + new_recs

with open(dataset_path, 'w') as f:
    for r in merged_recs:
        f.write(json.dumps(r) + '\n')

vendor_counts = Counter(r['vendor'] for r in merged_recs)
label_counts = Counter(l.split('-')[-1] for r in merged_recs for l in r['labels'])

stats = {
    "pages": len(merged_recs),
    "vendors": dict(vendor_counts),
    "label_counts": dict(label_counts),
}

with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"Merged dataset successfully saved: {len(merged_recs)} total pages across {len(vendor_counts)} distinct vendors.")
