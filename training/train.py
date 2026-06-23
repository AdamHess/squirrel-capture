"""
Train YOLO on auto-labeled squirrel data.

Usage:
  python training/train.py --data data/labeled/data.yaml --epochs 100

Run this on your Windows desktop (GTX 2080) after syncing labeled data
from the Linux server.
"""

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

log = logging.getLogger("train")


def create_data_yaml(label_dir, output_path, class_names=None):
    label_dir = Path(label_dir)
    images_dir = label_dir / "images"
    labels_dir = label_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        log.error(
            "Expected %s and %s to exist",
            images_dir,
            labels_dir,
        )
        return False

    images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not images:
        log.error("No images found in %s", images_dir)
        return False

    import random

    random.shuffle(images)
    split = int(len(images) * 0.8)

    train_imgs = images[:split]
    val_imgs = images[split:]

    train_dir = label_dir / "train" / "images"
    val_dir = label_dir / "val" / "images"
    train_lbl = label_dir / "train" / "labels"
    val_lbl = label_dir / "val" / "labels"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)

    import shutil

    for img in train_imgs:
        shutil.copy2(str(img), str(train_dir / img.name))
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(str(lbl), str(train_lbl / lbl.name))

    for img in val_imgs:
        shutil.copy2(str(img), str(val_dir / img.name))
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(str(lbl), str(val_lbl / lbl.name))

    classes = set()
    for lbl_file in labels_dir.glob("*.txt"):
        for line in lbl_file.read_text().strip().splitlines():
            if line.strip():
                classes.add(int(line.split()[0]))
    classes = sorted(classes)

    if class_names is None:
        class_names = {c: f"class_{c}" for c in classes}
        if 9 in classes:
            class_names[9] = "squirrel"

    data = {
        "train": str(train_dir.parent),
        "val": str(val_dir.parent),
        "nc": max(classes) + 1 if classes else 1,
        "names": class_names,
    }

    import yaml

    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    log.info(
        "Created %s with %d train / %d val images",
        output_path,
        len(train_imgs),
        len(val_imgs),
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Train YOLO on squirrel data")
    parser.add_argument("--data", default="data/labeled", help="Path to labeled data directory")
    parser.add_argument(
        "--model", default="yolo11n.pt", help="Base model (yolo11n.pt, yolo11s.pt, etc)"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="0", help="Device (0=cuda:0, cpu, etc.)")
    parser.add_argument("--project", default="runs", help="Output project directory")
    parser.add_argument("--name", default="squirrel", help="Experiment name")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    label_dir = Path(args.data)
    data_yaml = label_dir / "data.yaml"

    if not data_yaml.exists():
        log.info("No data.yaml found, creating from %s", label_dir)
        if not create_data_yaml(label_dir, data_yaml):
            return

    log.info("Loading base model: %s", args.model)
    model = YOLO(args.model)

    log.info(
        "Starting training: %d epochs, batch %d, imgsz %d, device %s",
        args.epochs,
        args.batch,
        args.imgsz,
        args.device,
    )

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=20,
        save=True,
        save_period=10,
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        augment=True,
        verbose=True,
    )

    log.info(
        "Training complete. Best model saved to %s/%s/weights/best.pt",
        args.project,
        args.name,
    )
    log.info("Results: %s", results)


if __name__ == "__main__":
    main()
