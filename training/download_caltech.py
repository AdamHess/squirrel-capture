"""
Download Caltech Camera Traps images for underrepresented classes.

Downloads raccoon, rodent, and squirrel images with bounding boxes
and converts to YOLO format.

Usage:
    uv run python training/download_caltech.py
"""

import json
import logging
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("download_caltech")

ANNOTATIONS_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "caltechcameratraps/labels/caltech_bboxes_20200316.json"
)
IMAGE_BASE_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "caltech-unzipped/cct_images/"
)

CLASS_MAP = {
    "raccoon": "Raccoon",
    "rodent": "Mouse",  # map rodent → Mouse class (closest match)
    "squirrel": "Squirrel",
}
# Class indices matching our existing 6-class system
OUR_CLASS_IDS = {
    "Squirrel": 0,
    "Raccoon": 2,
    "Mouse": 4,
}

# Target images PER CLASS to download from Caltech (to reach ~800 total per class)
# Current: Squirrel=393, Raccoon=85, Bird=800, Cat=800, Mouse=150, Person=800
DEFAULT_MAX_PER_CLASS = {
    "Squirrel": 410,   # 393 + 410 = 803
    "Raccoon": 720,    # 85 + 720 = 805
    "Mouse": 650,      # 150 + 650 = 800
}


def download_annotations(url: str) -> dict:
    log.info("Downloading annotations from %s", url)
    req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())


def download_image(url: str, dst_path: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "squirrel-capture/1.0"})
            resp = urllib.request.urlopen(req, timeout=30)
            with open(dst_path, "wb") as f:
                f.write(resp.read())
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                log.warning("Failed to download %s: %s", url, e)
    return False


def convert_to_yolo(bbox, img_width, img_height):
    """Convert COCO Camera Traps bbox [x, y, w, h] to YOLO [cx, cy, w, h] normalized."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_width
    cy = (y + h / 2) / img_height
    w = w / img_width
    h = h / img_height
    # Clamp to [0, 1]
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    w = max(0, min(1, w))
    h = max(0, min(1, h))
    return cx, cy, w, h


def main():
    import argparse
    import random

    parser = argparse.ArgumentParser(description="Download Caltech Camera Traps subset")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Override max images per class")
    args = parser.parse_args()

    random.seed(42)

    data = download_annotations(ANNOTATIONS_URL)

    # Build lookups
    cat_name_to_id = {c["name"]: c["id"] for c in data["categories"]}
    images = {img["id"]: img for img in data["images"]}

    # Build: category_id → caltech name (e.g. "raccoon")
    target_caltech_names = {
        cat_name_to_id[caltech_name]: caltech_name
        for caltech_name in CLASS_MAP
    }

    # Collect annotations per image
    img_by_cid = defaultdict(list)
    for ann in data["annotations"]:
        cid = ann["category_id"]
        if cid in target_caltech_names:
            img_by_cid[cid].append(ann["image_id"])

    # Shuffle and limit per class
    selected_ids = set()
    for cid, img_ids in img_by_cid.items():
        caltech_name = target_caltech_names[cid]
        max_count = args.max_per_class or DEFAULT_MAX_PER_CLASS.get(caltech_name, 400)
        random.shuffle(img_ids)
        for img_id in img_ids[:max_count]:
            selected_ids.add(img_id)

    # Build annotation list for selected images
    img_annotations = defaultdict(list)
    for ann in data["annotations"]:
        cid = ann["category_id"]
        if ann["image_id"] in selected_ids and cid in target_caltech_names:
            img_annotations[ann["image_id"]].append(ann)

    log.info(
        "Found %d images with target annotations (after limiting)",
        len(img_annotations),
    )

    # Setup output directories per class
    out_root = Path("data/caltech")
    for caltech_name, class_name in CLASS_MAP.items():
        (out_root / class_name / "images").mkdir(parents=True, exist_ok=True)
        (out_root / class_name / "labels").mkdir(parents=True, exist_ok=True)

    # Download and convert
    downloaded = 0
    errors = 0
    skipped = 0

    for img_id, anns in img_annotations.items():
        img_info = images[img_id]
        file_name = img_info["file_name"]
        img_width = img_info.get("width", 0)
        img_height = img_info.get("height", 0)

        # All annotations for this image should be the same class
        # (camera trap usually has one animal per image)
        cid = anns[0]["category_id"]
        caltech_name = target_caltech_names[cid]
        class_name = CLASS_MAP[caltech_name]

        img_url = IMAGE_BASE_URL + file_name
        dst_path = out_root / class_name / "images" / file_name

        if dst_path.exists():
            skipped += 1
        else:
            if download_image(img_url, dst_path):
                downloaded += 1
            else:
                errors += 1
                continue

        # Write YOLO label file
        label_lines = []
        for ann in anns:
            cx, cy, bw, bh = convert_to_yolo(
                ann["bbox"], img_width, img_height
            )
            label_lines.append(
                f"{OUR_CLASS_IDS[class_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            )

        label_path = (
            out_root / class_name / "labels" / f"{Path(file_name).stem}.txt"
        )
        label_path.write_text("\n".join(label_lines))

        if (downloaded + errors + skipped) % 500 == 0:
            log.info(
                "Progress: %d downloaded, %d skipped, %d errors",
                downloaded,
                skipped,
                errors,
            )

    log.info(
        "Done: %d downloaded, %d skipped, %d errors",
        downloaded,
        skipped,
        errors,
    )

    # Print summary
    for caltech_name, class_name in CLASS_MAP.items():
        img_dir = out_root / class_name / "images"
        count = len(list(img_dir.glob("*")))
        log.info("  %s: %d images", class_name, count)


if __name__ == "__main__":
    main()
