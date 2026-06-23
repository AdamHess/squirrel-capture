"""
Train YOLO on squirrel/backyard-animal data.

Auto-registers trained models in models/registry.yaml for version tracking.

Usage:
  # Standard training
  python training/train.py --data data/exports/nyc-backyard-v1/data.yaml --epochs 100

  # With nighttime augmentations pre-applied
  python training/train.py --data data/exports/nyc-backyard-night-v1/data.yaml --name nyc-backyard-night-v1

  # Finetune from a prelabeled model
  python training/train.py --model nyc-backyard-v1 --data data/labeled/data.yaml --name finetune-v1 --model-type finetuned
"""

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

log = logging.getLogger("train")


def resolve_model_name_or_path(model_str: str) -> str:
    from models.registry import resolve_model_path, list_models

    p = Path(model_str)
    if p.exists():
        return str(p.resolve())
    try:
        return resolve_model_path(model_str)
    except ValueError:
        available = list(list_models().keys())
        log.warning(
            "'%s' not found as file or registry entry. Available models: %s. Falling back to raw name.",
            model_str,
            available,
        )
        return model_str


def register_trained_model(name, model_path, args, results, data_yaml):
    import yaml
    from models.registry import register_model

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    classes = cfg.get("names", [])
    if isinstance(classes, dict):
        classes = [classes[i] for i in sorted(classes.keys())]

    metrics = {}
    if hasattr(results, "box"):
        metrics = {
            "mAP50": round(float(getattr(results.box, "map50", 0)), 4),
            "mAP50-95": round(float(getattr(results.box, "map", 0)), 4),
        }

    register_model(
        name=name,
        model_path=model_path,
        model_type=getattr(args, "model_type", "finetuned"),
        base_model=args.model,
        dataset=f"custom:{data_yaml}",
        classes=classes,
        num_images=None,
        metrics=metrics or None,
        notes=getattr(args, "notes", None),
    )


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
        "--model", default="yolo11n.pt", help="Base model path or registry name"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="0", help="Device (0=cuda:0, cpu, etc.)")
    parser.add_argument("--project", default="runs", help="Output project directory")
    parser.add_argument("--name", default="nyc-backyard", help="Experiment name")
    parser.add_argument(
        "--model-type",
        default="prelabeled",
        choices=["prelabeled", "finetuned"],
        help="prelabeled = trained on public data; finetuned = trained on custom capture data",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: auto)")
    parser.add_argument("--notes", default=None, help="Notes for the registry entry")
    parser.add_argument("--no-register", action="store_true", help="Skip model registry registration")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    model_path = resolve_model_name_or_path(args.model)
    log.info("Resolved model: %s -> %s", args.model, model_path)

    data_path = Path(args.data)
    if data_path.is_file():
        data_yaml = data_path
    else:
        data_yaml = data_path / "data.yaml"

    if not data_yaml.exists():
        log.info("No data.yaml found, creating from %s", data_path)
        if not create_data_yaml(data_path, data_yaml):
            return

    log.info("Loading base model: %s", model_path)
    model = YOLO(model_path)

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
        lr0=args.lr,
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

    best_pt = Path(args.project) / "detect" / args.name / "weights" / "best.pt"
    log.info("Training complete. Best model: %s", best_pt)

    if not args.no_register and best_pt.exists():
        register_trained_model(args.name, best_pt, args, results, data_yaml)

    log.info("Results: %s", results)


if __name__ == "__main__":
    main()
