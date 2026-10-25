import base64

with open('data/gst_smoke_dataset.jsonl', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('ascii')

with open('gst_finetune_smoke.py', 'r') as f:
    code = f.read()

# Update holdout vendors to include AMULGCMMF
code = code.replace(
    'HOLDOUT_VENDORS = "27AFVFS8057B1ZG,29AAICA3918J1ZE,27AAECB1611P1ZN"',
    'HOLDOUT_VENDORS = "27AFVFS8057B1ZG,29AAICA3918J1ZE,27AAECB1611P1ZN,AMULGCMMF"'
)

# Replace _DATASET_B64 = "" with the actual base64
embedded = code.replace('_DATASET_B64 = ""', f'_DATASET_B64 = "{b64}"')

out_path = '/tmp/gst_finetune_smoke_embedded_v8.py'
with open(out_path, 'w') as f:
    f.write(embedded)

print(f"Generated {out_path}, size: {len(embedded)} bytes.")
