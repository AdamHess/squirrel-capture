"""Merge v8 training set (152) + new hand-labeled frames (116) → squirrel-v9."""
import shutil
import random
from pathlib import Path

V8_DIR = Path("data/exports/squirrel-v8")
NEW_DIR = Path("data/exports/hand-label-0626")
V9_DIR = Path("data/exports/squirrel-v9")

random.seed(42)

# Clean/create v9 dirs
for d in ["train/images", "train/labels", "val/images", "val/labels"]:
    (V9_DIR / d).mkdir(parents=True, exist_ok=True)
    for f in (V9_DIR / d).iterdir():
        f.unlink()

# Collect v8 images + labels
v8_samples = []
for split in ["train", "val"]:
    for f in (V8_DIR / split / "images").glob("*.jpg"):
        label = V8_DIR / split / "labels" / (f.stem + ".txt")
        if label.exists():
            v8_samples.append((f, label))
print(f"v8 samples: {len(v8_samples)}")

# Collect new hand-labeled samples
new_samples = []
for f in (NEW_DIR / "images").glob("*.jpg"):
    label = NEW_DIR / "labels" / (f.stem + ".txt")
    if label.exists():
        # Verify label is non-empty
        if label.stat().st_size > 0:
            new_samples.append((f, label))
print(f"New hand-labeled samples: {len(new_samples)}")

# Combine and split 80/20
all_samples = v8_samples + new_samples
random.shuffle(all_samples)
split_idx = int(len(all_samples) * 0.8)
train = all_samples[:split_idx]
val = all_samples[split_idx:]

print(f"Total: {len(all_samples)}  Train: {len(train)}  Val: {len(val)}")

for split_name, samples in [("train", train), ("val", val)]:
    for img_path, label_path in samples:
        shutil.copy2(img_path, V9_DIR / split_name / "images" / img_path.name)
        shutil.copy2(label_path, V9_DIR / split_name / "labels" / label_path.name)

# Write data.yaml
data_yaml = (
    f"names:\n"
    f"  - squirrel\n"
    f"nc: 1\n"
    f"train: {(V9_DIR / 'train').resolve()}\n"
    f"val: {(V9_DIR / 'val').resolve()}\n"
)
(V9_DIR / "data.yaml").write_text(data_yaml)
print(f"\nData YAML: {V9_DIR / 'data.yaml'}")
print("Done!")
