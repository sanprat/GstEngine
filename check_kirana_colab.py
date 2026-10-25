import subprocess, sys, json

try:
    import datasets
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
    import datasets

from datasets import load_dataset

print("Loading build-small-hackathon/kirana-invoice-train-data...")
ds = load_dataset("build-small-hackathon/kirana-invoice-train-data")
print("Dataset structure:")
print(ds)

train_ds = ds["train"]
print(f"\nTrain size: {len(train_ds)}")

for i in range(min(5, len(train_ds))):
    sample = train_ds[i]
    img = sample["image"]
    resp = json.loads(sample["response"])
    print(f"\n--- Sample {i} ---")
    print(f"Image size: {img.size}, mode: {img.mode}")
    print(f"Supplier: {resp.get('supplier')} | GSTIN: {resp.get('gstin_supplier')}")
    print(f"Buyer: {resp.get('buyer')} | GSTIN: {resp.get('gstin_buyer')}")
    print(f"Invoice No: {resp.get('invoice_number')} | Date: {resp.get('date')}")
    print(f"Line items count: {len(resp.get('line_items', []))}")
    if resp.get('line_items'):
        print(f"  First item: {resp['line_items'][0]}")
    print(f"Subtotal: {resp.get('subtotal')}, GST: {resp.get('gst_total')}, Total: {resp.get('invoice_total')}")
