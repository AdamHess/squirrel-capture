"""Extract motion frames via KNN + CLAHE lighting normalization for garden scenes."""
import cv2
import numpy as np
import os
from pathlib import Path

VIDEO_DIR = r"C:\Users\Adam\Desktop\Backyard_2026-06-27_0730-0930"
OUTPUT = Path(r"C:\Users\Adam\squirrel-capture\data\exports\motion-extract-0627\images")
OUTPUT.mkdir(parents=True, exist_ok=True)

AREA_MIN = 150               # smaller = more sensitive to distant motion
KNN_THRESH = 300             # lower = more sensitive
DEDUP = 15
PROC_W = 640
BASE_LR = 0.005

clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

videos = sorted([os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])
existing = set(f.name for f in OUTPUT.glob("*.jpg"))
print(f"Videos: {len(videos)}, Existing: {len(existing)}")

total_added = 0

for vi, vpath in enumerate(videos):
    vname = os.path.basename(vpath)
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = PROC_W / W
    proc_h = int(H * scale)

    knn = cv2.createBackgroundSubtractorKNN(history=200, dist2Threshold=KNN_THRESH, detectShadows=True)

    last_cx, last_cy = None, None
    seg_saved = 0
    fi = 0
    avg_bright = 0
    vid_added = 0

    while fi < total_f:
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1

        # CLAHE normalization for lighting invariance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_norm = clahe.apply(gray)
        small = cv2.resize(gray_norm, (PROC_W, proc_h))

        # Adaptive learning rate based on brightness change
        bright = np.mean(small)
        delta = abs(bright - avg_bright) if avg_bright else 0
        avg_bright = avg_bright * 0.95 + bright * 0.05
        lr = min(BASE_LR * max(delta, 1), 0.03)

        # KNN foreground
        fg = knn.apply(small, learningRate=lr)
        fg = cv2.medianBlur(fg, 5)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        # Find motion
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion = [c for c in contours if cv2.contourArea(c) > AREA_MIN]

        if motion:
            largest = max(motion, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                d = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2) if last_cx is not None else DEDUP + 1

                if d > DEDUP:
                    fname = f"mv_{vi+1:02d}_{fi:06d}.jpg"
                    out = OUTPUT / fname
                    if not out.exists():
                        cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        seg_saved += 1
                        vid_added += 1
                        last_cx, last_cy = cx, cy
                        if seg_saved == 1:
                            print(f"  [{vi+1}] {vname[:20]} motion F{fi}", flush=True)
        else:
            if seg_saved > 0:
                print(f"  -> {seg_saved} frames", flush=True)
                seg_saved = 0
                last_cx, last_cy = None, None

    if seg_saved > 0:
        print(f"  -> {seg_saved} frames", flush=True)
    cap.release()
    if vid_added:
        print(f"  [{vi+1}] +{vid_added} new", flush=True)
    total_added += vid_added

print(f"\nDone! +{total_added} new frames. Total: {len(list(OUTPUT.glob('*.jpg')))}")
