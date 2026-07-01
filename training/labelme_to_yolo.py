"""Convert LabelMe JSON annotations to YOLO format and prepare dataset."""
import json
import shutil
import random
from pathlib import Path

SRC = Path(r"C:\Users\Adam\squirrel-capture\data\exports\motion-extract-0627\images")
OUT = Path(r"C:\Users\Adam\squirrel-capture\data\exports\motion-extract-ds")
random.seed(42)

# Collect labeled images
labeled = []
for f in SRC.glob("*.json"):
    img_stem = f.stem  # same name as the image
    img_path = SRC / f"{img_stem}.jpg"
    if not img_path.exists():
        continue
    with open(f) as fh:
        data = json.load(fh)

    boxes = []
    img_w = data.get("imageWidth", 2560)
    img_h = data.get("imageHeight", 1920)
    for shape in data.get("shapes", []):
        if shape["shape_type"] != "rectangle":
            continue
        pts = shape["points"]
        # LabelMe stores [top-left, bottom-right]
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        # Convert to YOLO normalized cx, cy, w, h
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = abs(x2 - x1) / img_w
        bh = abs(y2 - y1) / img_h
        boxes.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if boxes:
        labeled.append((img_path, boxes))

print(f"Labeled images: {len(labeled)} ({sum(len(b) for _, b in labeled)} boxes)")

# Split 80/20
random.shuffle(labeled)
split = int(len(labeled) * 0.8)
train = labeled[:split]
val = labeled[split:]

for split_name, samples in [("train", train), ("val", val)]:
    (OUT / split_name / "images").mkdir(parents=True, exist_ok=True)
    (OUT / split_name / "labels").mkdir(parents=True, exist_ok=True)
    for img_path, boxes in samples:
        shutil.copy2(img_path, OUT / split_name / "images" / img_path.name)
        label_path = OUT / split_name / "labels" / f"{img_path.stem}.txt"
        label_path.write_text("\n".join(boxes))

# data.yaml
data_yaml = f"names:\n  - squirrel\nnc: 1\ntrain: {(OUT / 'train').resolve()}\nval: {(OUT / 'val').resolve()}\n"
(OUT / "data.yaml").write_text(data_yaml)

print(f"Train: {len(train)}  Val: {len(val)}")
print(f"Data YAML: {OUT / 'data.yaml'}")
print("Done!")
