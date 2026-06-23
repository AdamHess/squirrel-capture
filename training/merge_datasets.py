"""
Merge OpenImages + Caltech data into a balanced dataset (~800 per class),
then fine-tune the nyc-backyard-v2 model on the merged data.

Usage:
    uv run python training/merge_datasets.py
"""

import random
import shutil
from pathlib import Path

import yaml

random.seed(42)

# Source directories
OPENIMAGES_DIR = Path("data/openimages")
CALTECH_DIR = Path("data/caltech")
OUTPUT_DIR = Path("data/exports/nyc-backyard-v3")

# Target images per class (800 each)
TARGET_PER_CLASS = 800

# Class mapping (name → id)
CLASSES = ["Squirrel", "Bird", "Raccoon", "Cat", "Mouse", "Person"]

# Source: Caltech class → our class index + max to include
CALTECH_SOURCES = {
    "Squirrel": {"class": "Squirrel", "max": TARGET_PER_CLASS - 393},
    "Raccoon": {"class": "Raccoon", "max": TARGET_PER_CLASS - 85},
    "Mouse": {"class": "Mouse", "max": TARGET_PER_CLASS - 150},
}


def collect_openimages():
    """Collect all OpenImages samples, return [(img_path, class_id), ...]."""
    items = []
    for cls_name in CLASSES:
        cls_dir = OPENIMAGES_DIR / cls_name.lower()
        img_dir = cls_dir / "images"
        lbl_dir = cls_dir / "darknet"
        class_id = CLASSES.index(cls_name)

        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                items.append({
                    "img_path": img_path,
                    "lbl_path": lbl_path,
                    "class_id": class_id,
                    "class_name": cls_name,
                })

    return items


def collect_caltech():
    """Collect Caltech samples with correct class mapping."""
    items = []
    for caltech_dir, info in CALTECH_SOURCES.items():
        src_img_dir = CALTECH_DIR / info["class"] / "images"
        src_lbl_dir = CALTECH_DIR / info["class"] / "labels"
        class_id = CLASSES.index(info["class"])

        if not src_img_dir.exists():
            continue

        for img_path in sorted(src_img_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
            # Only include if the label has our class_id
            # (caltech labels use hardcoded class IDs from OUR_CLASS_IDS)
            if lbl_path.exists():
                items.append({
                    "img_path": img_path,
                    "lbl_path": lbl_path,
                    "class_id": class_id,
                    "class_name": info["class"],
                    "source": "caltech",
                })

    return items


def merge_and_balance():
    """Merge datasets, balance to ~800 per class, create train/val split."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all items
    oi_items = collect_openimages()
    cal_items = collect_caltech()

    print(f"OpenImages: {len(oi_items)} items")
    for c in CLASSES:
        count = sum(1 for i in oi_items if i["class_name"] == c)
        print(f"  {c}: {count}")
    print(f"Caltech: {len(cal_items)} items")
    for c in ["Squirrel", "Raccoon", "Mouse"]:
        count = sum(1 for i in cal_items if i["class_name"] == c)
        print(f"  {c}: {count}")

    # Group by class
    by_class = {c: [] for c in CLASSES}
    for item in oi_items + cal_items:
        by_class[item["class_name"]].append(item)

    # Subsample each class to TARGET_PER_CLASS
    balanced = []
    for cls_name, items in by_class.items():
        random.shuffle(items)
        selected = items[:TARGET_PER_CLASS]
        balanced.extend(selected)
        print(f"{cls_name}: {len(items)} total -> {len(selected)} selected")

    random.shuffle(balanced)
    print(f"\nTotal balanced dataset: {len(balanced)} images")

    # Split 80/20
    split_idx = int(len(balanced) * 0.8)
    train_items = balanced[:split_idx]
    val_items = balanced[split_idx:]

    print(f"Train: {len(train_items)}, Val: {len(val_items)}")

    # Copy to output directories
    for split_name, items in [("train", train_items), ("val", val_items)]:
        img_dir = OUTPUT_DIR / split_name / "images"
        lbl_dir = OUTPUT_DIR / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for item in items:
            stem = item["img_path"].stem
            ext = item["img_path"].suffix
            dst_img = img_dir / f"{stem}{ext}"
            dst_lbl = lbl_dir / f"{stem}.txt"

            # Only copy if not already there (dedup by filename)
            if not dst_img.exists():
                shutil.copy2(str(item["img_path"]), str(dst_img))

            # Read label, remap class_id to match our index
            lines = []
            for line in item["lbl_path"].read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    parts[0] = str(item["class_id"])
                    lines.append(" ".join(parts))
            dst_lbl.write_text("\n".join(lines))

    # Create data.yaml
    data_yaml = {
        "train": str((OUTPUT_DIR / "train").resolve()),
        "val": str((OUTPUT_DIR / "val").resolve()),
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"data.yaml -> {yaml_path}")

    return yaml_path


if __name__ == "__main__":
    merge_and_balance()
