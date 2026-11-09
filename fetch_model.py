#!/usr/bin/env python3
"""
fetch_model.py — Download the v2.0 Desktop LayoutLM INT8 ONNX model from GitHub Releases.
"""

import os
import sys
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "desktop_models"
MODEL_FILE = MODEL_DIR / "model_desktop_int8.onnx"
RELEASE_TAG = "v2.0.0-desktop-95"
MODEL_URL = f"https://github.com/sanprat/GstEngine/releases/download/{RELEASE_TAG}/model_desktop_int8.onnx"
EXPECTED_MIN_BYTES = 100 * 1024 * 1024  # ~107.66 MB


def report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading model_desktop_int8.onnx: {mb:.1f} / {total_mb:.1f} MB ({percent:.1f}%)")
        sys.stdout.flush()


def download_model(force=False):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_FILE.exists() and not force:
        size_mb = MODEL_FILE.stat().st_size / (1024 * 1024)
        if MODEL_FILE.stat().st_size >= EXPECTED_MIN_BYTES:
            print(f"✅ Desktop model already present: {MODEL_FILE} ({size_mb:.2f} MB)")
            return True
        else:
            print(f"⚠️ Model file appears incomplete ({size_mb:.2f} MB). Re-downloading...")

    print(f"📥 Fetching v2.0 Desktop LayoutLM INT8 ONNX Model ({RELEASE_TAG})...")
    print(f"Source: {MODEL_URL}")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE, reporthook=report_progress)
        print("\n✅ Download complete!")
        size_mb = MODEL_FILE.stat().st_size / (1024 * 1024)
        print(f"Saved: {MODEL_FILE} ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"\n❌ Failed to download model: {e}")
        if MODEL_FILE.exists():
            MODEL_FILE.unlink()
        return False


if __name__ == "__main__":
    force_download = "--force" in sys.argv
    success = download_model(force=force_download)
    sys.exit(0 if success else 1)
