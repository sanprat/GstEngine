import os, sys, json, tarfile, subprocess
from PIL import Image

tar_path = "/content/images_8.tar.gz"
img_dir = "/content/images_8"
os.makedirs(img_dir, exist_ok=True)

if os.path.exists(tar_path):
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=img_dir)
    print(f"Extracted {len(os.listdir(img_dir))} images.")

try:
    import easyocr
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "easyocr"])
    import easyocr

image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
print(f"Processing {len(image_files)} images with EasyOCR (GPU)...")

reader = easyocr.Reader(["en"], gpu=True)
all_ocr = {}

for img_name in image_files:
    img_path = os.path.join(img_dir, img_name)
    im = Image.open(img_path)
    w_img, h_img = im.size
    print(f"\nProcessing {img_name} ({w_img}x{h_img})...")
    
    results = reader.readtext(img_path)
    print(f"  Detected blocks: {len(results)}")
    
    out_words = []
    for bbox, text, prob in results:
        text = text.strip()
        if not text:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        words = text.split()
        if len(words) <= 1:
            bx0 = max(0, min(1000, int(round(min_x * 1000.0 / w_img))))
            by0 = max(0, min(1000, int(round(min_y * 1000.0 / h_img))))
            bx1 = max(bx0 + 1, min(1000, int(round(max_x * 1000.0 / w_img))))
            by1 = max(by0 + 1, min(1000, int(round(max_y * 1000.0 / h_img))))
            out_words.append({"text": text, "box": [bx0, by0, bx1, by1], "prob": float(prob)})
        else:
            total_chars = sum(len(w) for w in words)
            cur_x = min_x
            box_w = max_x - min_x
            for w in words:
                w_len = len(w)
                char_ratio = w_len / max(1, total_chars)
                word_w = box_w * char_ratio
                bx0 = max(0, min(1000, int(round(cur_x * 1000.0 / w_img))))
                by0 = max(0, min(1000, int(round(min_y * 1000.0 / h_img))))
                bx1 = max(bx0 + 1, min(1000, int(round((cur_x + word_w) * 1000.0 / w_img))))
                by1 = max(by0 + 1, min(1000, int(round(max_y * 1000.0 / h_img))))
                out_words.append({"text": w, "box": [bx0, by0, bx1, by1], "prob": float(prob)})
                cur_x += word_w
                
    all_ocr[img_name] = {
        "words": [ow["text"] for ow in out_words],
        "boxes": [ow["box"] for ow in out_words],
        "probs": [ow["prob"] for ow in out_words],
        "w": w_img,
        "h": h_img
    }
    print(f"  Extracted {len(out_words)} words.")

out_json = "/content/ocr_batch_8.json"
with open(out_json, "w") as f:
    json.dump(all_ocr, f)
print(f"\nAll OCR results written to {out_json} ({os.path.getsize(out_json)} bytes).")
