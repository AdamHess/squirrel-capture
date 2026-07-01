import random
import shutil
from pathlib import Path

import yaml


def prepare_dataset(src_dir: str, out_dir: str, split=(0.8, 0.2)):
    src = Path(src_dir)
    out = Path(out_dir)

    imgs = sorted((src / "images" / "train").glob("*.jpeg"))
    imgs += sorted((src / "images" / "train").glob("*.jpg"))
    imgs += sorted((src / "images" / "train").glob("*.png"))

    if not imgs:
        print("No images found in", src / "images" / "train")
        return

    random.shuffle(imgs)
    n = len(imgs)
    split_idx = int(n * split[0])

    for split_name, img_list in [("train", imgs[:split_idx]), ("val", imgs[split_idx:])]:
        img_dir = out / split_name / "images"
        lbl_dir = out / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in img_list:
            shutil.copy2(str(img), str(img_dir / img.name))
            lbl = src / "labels" / "train" / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(str(lbl), str(lbl_dir / f"{img.stem}.txt"))

    data_yaml = {
        "train": str((out / "train").resolve()),
        "val": str((out / "val").resolve()),
        "nc": 2,
        "names": ["Squirrel", "Nut"],
    }

    with open(out / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    print(f"Split {n} images: {split_idx} train, {n - split_idx} val")
    print(f"data.yaml at {out / 'data.yaml'}")
    print(f"Train images: {len(list((out / 'train' / 'images').iterdir()))}")
    print(f"Val images: {len(list((out / 'val' / 'images').iterdir()))}")


if __name__ == "__main__":
    prepare_dataset(
        src_dir="data/squirrelsandnuts/squirrelsandnuts_train",
        out_dir="data/exports/squirrelsandnuts",
    )
