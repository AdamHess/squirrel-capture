"""Merge refined (high conf) with original (low conf) labels."""
import json
import shutil
import random
from pathlib import Path

SRC = Path("data/exports/all-hand-labeled")
OUT = Path("data/exports/master-ds-v2")
CONF_THRESH = 0.60

refined_dir = SRC / "refined-labels"
orig_dir = SRC / "labels"
img_dir = SRC / "images"
images = sorted(img_dir.glob("*.jpg"))

# Load confidence metadata
meta_path = SRC / "refined-meta.json"
confs = json.loads(meta_path.read_text()) if meta_path.exists() else {}

random.seed(42)
random.shuffle(images)
n_train = int(len(images) * 0.8)
n_val = int(len(images) * 0.9)
splits = [("train", images[:n_train]), ("val", images[n_train:n_val]), ("test", images[n_val:])]

for split_name, _ in splits:
    for d in [OUT / split_name / "images", OUT / split_name / "labels"]:
        if d.exists():
            for f in d.iterdir(): f.unlink()
        d.mkdir(parents=True, exist_ok=True)

refined_count = 0
original_count = 0

for split_name, split_imgs in splits:
    for img_path in split_imgs:
        stem = img_path.stem
        ref_path = refined_dir / f"{stem}.txt"

        use_refined = confs.get(stem, 0) >= CONF_THRESH and ref_path.exists()

        if use_refined:
            shutil.copy2(ref_path, OUT / split_name / "labels" / f"{stem}.txt")
            refined_count += 1
        else:
            orig_path = orig_dir / f"{stem}.txt"
            if orig_path.exists():
                shutil.copy2(orig_path, OUT / split_name / "labels" / f"{stem}.txt")
                original_count += 1
            else:
                continue

        shutil.copy2(img_path, OUT / split_name / "images" / img_path.name)

yaml = f"names:\n  - squirrel\nnc: 1\ntrain: {(OUT / 'train').resolve()}\nval: {(OUT / 'val').resolve()}\n"
(OUT / "data.yaml").write_text(yaml)

print(f"Refined (conf >= {CONF_THRESH}): {refined_count}")
print(f"Original:                       {original_count}")
print(f"Total:                          {refined_count + original_count}")
