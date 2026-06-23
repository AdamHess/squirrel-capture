import random
import shutil
from pathlib import Path

import yaml

CLASSES = ["Squirrel", "Bird", "Raccoon", "Cat", "Mouse", "Person"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def consolidate(src_dir: str, out_dir: str, split=(0.8, 0.2), seed=42):
    src = Path(src_dir)
    out = Path(out_dir)

    random.seed(seed)

    all_items = []
    for cls_name in CLASSES:
        cls_dir = src / cls_name.lower()
        img_dir = cls_dir / "images"
        lbl_dir = cls_dir / "darknet"
        class_id = CLASS_TO_IDX[cls_name]

        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                all_items.append((img_path, lbl_path, class_id))
            else:
                print(f"Warning: no label for {img_path}")

    random.shuffle(all_items)
    n = len(all_items)
    split_idx = int(n * split[0])
    print(f"Total images: {n}")

    for split_name, items in [("train", all_items[:split_idx]), ("val", all_items[split_idx:])]:
        img_dir = out / split_name / "images"
        lbl_dir = out / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path, class_id in items:
            stem = img_path.stem
            dst_img = img_dir / f"{stem}{img_path.suffix}"
            shutil.copy2(str(img_path), str(dst_img))

            lines = []
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    parts[0] = str(class_id)
                    lines.append(" ".join(parts))
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))

        print(f"{split_name}: {len(items)} images")

    data_yaml = {
        "train": str((out / "train").resolve()),
        "val": str((out / "val").resolve()),
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    yaml_path = out / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"data.yaml -> {yaml_path}")


if __name__ == "__main__":
    consolidate(
        src_dir="data/openimages",
        out_dir="data/exports/nyc-backyard-v2",
    )
