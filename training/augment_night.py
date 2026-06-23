"""
Nighttime simulation augmentations for training YOLO on low-light backyard data.

Applies gamma darkening, Gaussian noise, contrast reduction, and random
brightness/contrast shifts to simulate nighttime / dusk conditions.

Usage:
    uv run python training/augment_night.py \\
        --src data/exports/nyc-backyard-v1 \\
        --dst data/exports/nyc-backyard-night \\
        --factor 0.5   # fraction of images to augment
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

AUG_TAG = "_night"


def augment_image(img: np.ndarray, severity: float = 1.0) -> np.ndarray:
    h, w = img.shape[:2]

    # 1. Gamma darkening (random gamma 0.3-0.8)
    gamma = random.uniform(0.3, 0.8) * severity
    gamma = max(0.1, min(gamma, 1.0))
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    img = cv2.LUT(img, table)

    # 2. Gaussian noise
    noise_std = random.uniform(5, 20) * severity
    noise = np.random.normal(0, noise_std, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype("uint8")

    # 3. Random brightness/contrast shift
    alpha = random.uniform(0.6, 1.0)
    beta = random.randint(-30, 0)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=int(beta * severity))

    # 4. Slight blur to simulate longer exposure / motion
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img


def augment_dataset(src_dir: str, dst_dir: str, factor: float = 0.5, severity: float = 1.0):
    src = Path(src_dir)
    dst = Path(dst_dir)
    random.seed(42)
    np.random.seed(42)

    for split in ("train", "val"):
        src_img_dir = src / split / "images"
        src_lbl_dir = src / split / "labels"
        dst_img_dir = dst / split / "images"
        dst_lbl_dir = dst / split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(src_img_dir.glob("*"))
        random.shuffle(images)
        aug_count = int(len(images) * factor)

        for img_path in images:
            stem = img_path.stem
            ext = img_path.suffix
            dst_img_path = dst_img_dir / f"{stem}{ext}"
            dst_lbl_path = dst_lbl_dir / f"{stem}.txt"

            lbl_path = src_lbl_dir / f"{stem}.txt"
            if lbl_path.exists():
                shutil.copy2(str(lbl_path), str(dst_lbl_path))

            should_aug = aug_count > 0
            if should_aug:
                aug_count -= 1

            if should_aug:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = augment_image(img, severity)
                    cv2.imwrite(str(dst_img_path), img)
                    aug_stem = f"{stem}{AUG_TAG}"
                    aug_img_path = dst_img_dir / f"{aug_stem}{ext}"
                    shutil.copy2(str(lbl_path), dst_lbl_dir / f"{aug_stem}.txt")
                    cv2.imwrite(str(aug_img_path), img)
            else:
                shutil.copy2(str(img_path), str(dst_img_path))

        print(f"{split}: {len(images)} originals + {int(len(images) * factor)} nighttime augs")

    shutil.copy2(str(src / "data.yaml"), str(dst / "data.yaml"))
    print(f"Dataset saved to {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add nighttime augmentations to dataset")
    parser.add_argument("--src", default="data/exports/nyc-backyard-v1", help="Source dataset")
    parser.add_argument("--dst", default="data/exports/nyc-backyard-night-v1", help="Output dataset")
    parser.add_argument("--factor", type=float, default=0.5, help="Fraction of images to augment (0-1)")
    parser.add_argument("--severity", type=float, default=1.0, help="Severity multiplier (0-2)")
    args = parser.parse_args()
    augment_dataset(args.src, args.dst, args.factor, args.severity)
