"""
Download specific classes from OpenImages v7 using FiftyOne and export to YOLO format.

Usage:
    uv run python training/download_openimages.py --max-per-class 300
"""
import argparse
import logging
import warnings
from pathlib import Path

log = logging.getLogger("download_oi")


def download_classes(
    classes: list[str],
    output_dir: str = "data/openimages",
    max_per_class: int = 300,
    force: bool = False,
):
    out = Path(output_dir)
    data_yaml_path = out / "data.yaml"

    if data_yaml_path.exists() and not force:
        log.info("Dataset already exists at %s (use --force to re-download)", out)
        return str(data_yaml_path)

    import fiftyone as fo
    import fiftyone.zoo as foz

    fo.config.dataset_zoo_dir = str(out / "fiftyone_cache")

    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    all_images = []
    all_labels = []

    for cls in classes:
        log.info("Downloading '%s' (max %d samples)...", cls, max_per_class)

        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=[cls],
            max_samples=max_per_class,
            shuffle=True,
        )

        for sample in dataset:
            img_path = sample.filepath
            detections = sample.ground_truth.detections
            labels = []
            for det in detections:
                if det.label in class_to_idx:
                    x, y, w, h = det.bounding_box
                    labels.append({
                        "class_id": class_to_idx[det.label],
                        "class_name": det.label,
                        "bbox": [x, y, w, h],
                    })
            if labels:
                all_images.append((img_path, labels))

        log.info("  Got %d images with '%s'", sum(1 for _, lbls in all_images if any(lbl["class_name"] == cls for lbl in lbls)), cls)

    import random
    random.shuffle(all_images)

    split_idx = int(len(all_images) * 0.8)
    splits = [("train", all_images[:split_idx]), ("val", all_images[split_idx:])]

    for split_name, img_list in splits:
        img_dir = out / split_name / "images"
        lbl_dir = out / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, labels in img_list:
            import shutil
            stem = Path(img_path).stem
            dst_img = img_dir / f"{stem}.jpg"
            shutil.copy2(img_path, str(dst_img))

            from PIL import Image
            with Image.open(img_path) as img:
                w, h = img.size
            lines = []
            for lbl in labels:
                x, y, bw, bh = lbl["bbox"]
                cx = x + bw / 2
                cy = y + bh / 2
                lines.append(f"{lbl['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))

    import yaml
    data_yaml = {
        "train": str((out / "train").resolve()),
        "val": str((out / "val").resolve()),
        "nc": len(classes),
        "names": classes,
    }

    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    log.info(
        "Downloaded %d images (%d train, %d val) for classes: %s",
        len(all_images),
        split_idx,
        len(all_images) - split_idx,
        classes,
    )

    return str(data_yaml_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OpenImages classes for NYC backyard animals")
    parser.add_argument("--max-per-class", type=int, default=300, help="Max images per class")
    parser.add_argument("--output", default="data/openimages", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Re-download if exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    classes = ["Squirrel", "Bird", "Raccoon", "Cat", "Rat", "Mouse", "Person"]

    download_classes(
        classes=classes,
        output_dir=args.output,
        max_per_class=args.max_per_class,
        force=args.force,
    )
