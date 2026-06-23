"""
Export labeled dataset for training:
  - Split into train/val sets
  - Create data.yaml
  - Optionally zip for transfer to training machine

Usage:
  python training/export_dataset.py --input data/labeled --output data/exports/my_export
"""
import argparse
import logging
import random
import shutil
from pathlib import Path

import yaml

log = logging.getLogger("export")


def export_dataset(input_dir, output_dir, split=(0.8, 0.2, 0.0),
                   class_names=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        log.error("Input must have images/ and labels/ subdirectories")
        return False

    images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    random.shuffle(images)

    n = len(images)
    n_train = int(n * split[0])
    n_val = int(n * split[1])

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]

    for split_name, img_list in [("train", train_imgs), ("val", val_imgs)]:
        out_img_dir = output_dir / split_name / "images"
        out_lbl_dir = output_dir / split_name / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in img_list:
            shutil.copy2(str(img), str(out_img_dir / img.name))
            lbl = labels_dir / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(str(lbl), str(out_lbl_dir / lbl.name))

    class_ids = set()
    for lbl_file in labels_dir.glob("*.txt"):
        for line in lbl_file.read_text().strip().splitlines():
            if line.strip():
                class_ids.add(int(line.split()[0]))

    if class_names is None:
        class_names = {}
        if 9 in class_ids:
            class_names[9] = "squirrel"
        for c in sorted(class_ids):
            if c not in class_names:
                class_names[c] = f"class_{c}"

    data_yaml = {
        "train": str((output_dir / "train").resolve()),
        "val": str((output_dir / "val").resolve()),
        "nc": max(class_ids) + 1 if class_ids else 1,
        "names": class_names,
    }

    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    log.info("Exported %d train, %d val images to %s", len(train_imgs), len(val_imgs), output_dir)
    log.info("Classes: %s", class_names)
    return True


def main():
    parser = argparse.ArgumentParser(description="Export labeled dataset")
    parser.add_argument("--input", default="data/labeled", help="Labeled dataset directory")
    parser.add_argument("--output", default="data/exports/dataset", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    export_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
