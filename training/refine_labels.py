"""Use master-v1 to generate tighter boxes on all hand-labeled images."""
import sys; sys.path.insert(0, '.')
import cv2
from pathlib import Path
from ultralytics import YOLO

SRC = Path("data/exports/all-hand-labeled")
REFINED = SRC / "refined-labels"
REFINED.mkdir(exist_ok=True)
model = YOLO("deploy/master-v1.pt")

images = sorted(SRC.glob("images/*.jpg"))
print(f"Processing {len(images)} images...")

updated = 0
for img_path in images:
    frame = cv2.imread(str(img_path))
    if frame is None:
        continue

    h, w = frame.shape[:2]
    r = model(frame, conf=0.25, verbose=False, device=0)[0]

    lines = []
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}")

    if lines:
        label_path = REFINED / f"{img_path.stem}.txt"
        label_path.write_text("\n".join(lines))
        updated += 1

    if (len(images) // 5) > 0 and (updated % (len(images) // 5)) == 0:
        pct = updated / len(images) * 100
        print(f"  {pct:.0f}%", flush=True)

print(f"\nDone! {updated}/{len(images)} refined labels -> {REFINED}")
