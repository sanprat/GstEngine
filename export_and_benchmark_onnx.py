import subprocess, sys

for pkg in ['onnx', 'onnxruntime', 'onnxscript']:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

import os, time, json
import numpy as np
import torch
from transformers import LayoutLMConfig, LayoutLMForTokenClassification
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

model_dir = "/content/gst_smoke"
print(f"Loading PyTorch model from {model_dir}...")
config = LayoutLMConfig.from_pretrained(model_dir)
model = LayoutLMForTokenClassification(config)
model.load_state_dict(torch.load(os.path.join(model_dir, "best_state.pt"), map_location="cpu"))
model.eval()

pt_size_mb = os.path.getsize(os.path.join(model_dir, "best_state.pt")) / (1024 * 1024)
print(f"PyTorch state_dict size: {pt_size_mb:.2f} MB")

seq_len = 768
dummy_input_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)
dummy_bbox = torch.zeros((1, seq_len, 4), dtype=torch.long)
dummy_bbox[:, :, 0] = 10; dummy_bbox[:, :, 1] = 10; dummy_bbox[:, :, 2] = 50; dummy_bbox[:, :, 3] = 30
dummy_attention_mask = torch.ones((1, seq_len), dtype=torch.long)
dummy_token_type_ids = torch.zeros((1, seq_len), dtype=torch.long)

# 1. Export FP32 ONNX
onnx_fp32_path = os.path.join(model_dir, "model_fp32.onnx")
print(f"Exporting to ONNX FP32: {onnx_fp32_path}...")
torch.onnx.export(
    model,
    (dummy_input_ids, dummy_bbox, dummy_attention_mask, dummy_token_type_ids),
    onnx_fp32_path,
    input_names=["input_ids", "bbox", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "bbox": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "token_type_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    },
    opset_version=14
)

fp32_size_mb = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
print(f"ONNX FP32 size: {fp32_size_mb:.2f} MB")

# 2. Dynamic INT8 Quantization
onnx_int8_path = os.path.join(model_dir, "model_int8.onnx")
print(f"Quantizing to ONNX INT8: {onnx_int8_path}...")
quantize_dynamic(
    onnx_fp32_path,
    onnx_int8_path,
    weight_type=QuantType.QInt8
)

int8_size_mb = os.path.getsize(onnx_int8_path) / (1024 * 1024)
print(f"ONNX INT8 size: {int8_size_mb:.2f} MB (Compression ratio: {fp32_size_mb / int8_size_mb:.2f}x)")

# 3. CPU Latency Benchmark (Mobile ARM / CPU simulation)
print("\nBenchmarking CPU inference latency (batch=1, seq_len=768)...")
session_fp32 = ort.InferenceSession(onnx_fp32_path, providers=["CPUExecutionProvider"])
session_int8 = ort.InferenceSession(onnx_int8_path, providers=["CPUExecutionProvider"])

inputs = {
    "input_ids": dummy_input_ids.numpy(),
    "bbox": dummy_bbox.numpy(),
    "attention_mask": dummy_attention_mask.numpy(),
    "token_type_ids": dummy_token_type_ids.numpy()
}

# Warmup
for _ in range(5):
    _ = session_fp32.run(None, inputs)
    _ = session_int8.run(None, inputs)

# Benchmark FP32
fp32_times = []
for _ in range(25):
    t0 = time.perf_counter()
    _ = session_fp32.run(None, inputs)
    fp32_times.append((time.perf_counter() - t0) * 1000)

# Benchmark INT8
int8_times = []
for _ in range(25):
    t0 = time.perf_counter()
    _ = session_int8.run(None, inputs)
    int8_times.append((time.perf_counter() - t0) * 1000)

avg_fp32 = np.mean(fp32_times)
avg_int8 = np.mean(int8_times)
speedup = avg_fp32 / avg_int8

print(f"FP32 CPU Latency: {avg_fp32:.1f} ms")
print(f"INT8 CPU Latency: {avg_int8:.1f} ms ({speedup:.2f}x speedup)")

# 4. Verify numerical parity (correlation of logits)
with torch.no_grad():
    pt_logits = model(dummy_input_ids, bbox=dummy_bbox, attention_mask=dummy_attention_mask, token_type_ids=dummy_token_type_ids).logits.numpy()
int8_logits = session_int8.run(None, inputs)[0]

pt_preds = np.argmax(pt_logits, axis=-1)
int8_preds = np.argmax(int8_logits, axis=-1)
token_agreement = np.mean(pt_preds == int8_preds) * 100

print(f"\nPrediction Parity:")
print(f"Token Prediction Agreement (PyTorch vs INT8 ONNX): {token_agreement:.2f}%")

summary = {
    "pt_size_mb": pt_size_mb,
    "fp32_size_mb": fp32_size_mb,
    "int8_size_mb": int8_size_mb,
    "avg_fp32_latency_ms": avg_fp32,
    "avg_int8_latency_ms": avg_int8,
    "speedup": speedup,
    "token_agreement_pct": token_agreement
}

with open(os.path.join(model_dir, "onnx_benchmark.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nBenchmark results written to onnx_benchmark.json!")
