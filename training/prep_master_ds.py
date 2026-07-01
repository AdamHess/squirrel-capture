"""Split 403 hand-labeled images into train/val for master training."""
import random
import shutil
from pathlib import Path

SRC = Path("data/exports/all-hand-labeled")
DST = Path("data/exports/master-ds")
random.seed(42)

# Collect all labeled images
samples = []
for f in sorted((SRC / "images").glob("*.jpg")):
    label = SRC / "labels" / (f.stem + ".txt")
    if label.exists() and label.stat().st_size > 0:
        samples.append((f, label))

print(f"Total labeled: {len(samples)}")

# Split 80/10/10 (train/val/test)
random.shuffle(samples)
n_train = int(len(samples) * 0.8)
n_val = int(len(samples) * 0.9)
train = samples[:n_train]
val = samples[n_train:n_val]
test = samples[n_val:]

print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

for split_name, samples_list in [("train", train), ("val", val), ("test", test)]:
    img_out = DST / split_name / "images"
    lbl_out = DST / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img_path, label_path in samples_list:
        shutil.copy2(img_path, img_out / img_path.name)
        shutil.copy2(label_path, lbl_out / label_path.name)

yaml = f"names:\n  - squirrel\nnc: 1\ntrain: {(DST / 'train').resolve()}\nval: {(DST / 'val').resolve()}\n"
(DST / "data.yaml").write_text(yaml)
print(f"\nData YAML: {DST / 'data.yaml'}")
print("Done!")
