"""Run YOLO inference on extracted frames and report why they're missed."""
import cv2
import sys
from pathlib import Path

from ultralytics import YOLO

model_path = "deploy/squirrel-v8.pt"
img_dir = Path("data/exports/hand-label-0626/images")
conf_threshold = 0.25
save_threshold = 0.30

model = YOLO(model_path)

results = []
files = sorted(img_dir.glob("*.jpg"))
print(f"Testing {len(files)} frames against {model_path} (conf ≥ {conf_threshold})...\n")

missed_at_25 = 0
missed_at_30 = 0
above_30 = 0
all_confs = []

for f in files:
    img = cv2.imread(str(f))
    if img is None:
        continue

    r = model(img, conf=0.01)[0]  # low threshold to see everything

    if len(r.boxes) == 0:
        results.append((f.name, None, None))
        missed_at_25 += 1
        missed_at_30 += 1
        continue

    confs = [box.conf[0].item() for box in r.boxes]
    max_conf = max(confs)
    all_confs.append(max_conf)

    if max_conf >= conf_threshold:
        if max_conf >= save_threshold:
            above_30 += 1
        else:
            missed_at_30 += 1
        results.append((f.name, max_conf, [round(c, 4) for c in confs]))
    else:
        results.append((f.name, max_conf, None))
        missed_at_25 += 1
        missed_at_30 += 1

print(f"Results:")
print(f"  Total frames:       {len(files)}")
print(f"  Detected ≥0.25:     {len(files) - missed_at_25} ({(len(files)-missed_at_25)/len(files)*100:.1f}%)")
print(f"  Detected ≥0.30:     {above_30} ({above_30/len(files)*100:.1f}%)")
print(f"  Missed (<0.25):     {missed_at_25} ({missed_at_25/len(files)*100:.1f}%)")
if all_confs:
    print(f"  Avg conf (if found): {sum(all_confs)/len(all_confs):.3f}")
    print(f"  Max conf:            {max(all_confs):.3f}")
    print(f"  Min conf:            {min(all_confs):.3f}")

print(f"\n=== Detailed results ===")
for name, conf, all_c in results:
    if conf is None:
        print(f"  {name}: NOTHING DETECTED")
    elif conf < conf_threshold:
        print(f"  {name}: conf={conf:.3f} (below {conf_threshold})")
    elif conf < save_threshold:
        print(f"  {name}: conf={conf:.3f} (between threshold and save)")
    else:
        print(f"  {name}: conf={conf:.3f} OK {all_c}")
