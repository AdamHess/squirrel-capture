"""Process remaining videos with more sensitive motion detection."""
import cv2
import numpy as np
import os
from pathlib import Path

VIDEO_DIR = r"C:\Users\Adam\Desktop\Backyard_2026-06-27_0730-0930"
OUTPUT = Path(r"C:\Users\Adam\squirrel-capture\data\exports\motion-extract-0627\images")
OUTPUT.mkdir(parents=True, exist_ok=True)

# More sensitive settings
MOTION_MIN_AREA = 150
MOG2_VAR = 20
DEDUP_DIST = 20
PROC_W = 640

# Only process videos 10+
videos = sorted([os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])
videos = videos[9:]  # index 9 = video 10
print(f"Remaining videos: {len(videos)} (10-20)")

# Also re-check videos 4-7 (no motion found)
recheck = [v for v in sorted([os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]) if "080414" in v or "081414" in v or "081915" in v or "082415" in v]
print(f"Re-checking {len(recheck)} zero-motion videos with lower threshold")

all_videos = videos + recheck
existing = set(OUTPUT.glob("*.jpg"))
before = len(existing)

for vi, vpath in enumerate(all_videos):
    vname = os.path.basename(vpath)
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = PROC_W / W

    mog2 = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=MOG2_VAR, detectShadows=True)
    last_cx, last_cy = None, None
    cooldown = 0
    seg_saved = 0
    fi = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1

        small = cv2.resize(frame, (PROC_W, int(H * scale)))
        fg = mog2.apply(small)
        fg = cv2.medianBlur(fg, 5)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in contours if cv2.contourArea(c) > MOTION_MIN_AREA]

        if big:
            largest = max(big, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2) if last_cx is not None else DEDUP_DIST + 1

                if dist > DEDUP_DIST:
                    fname = f"me_{vi+11:02d}_{fi:06d}.jpg"
                    outpath = OUTPUT / fname
                    if not outpath.exists():
                        cv2.imwrite(str(outpath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        seg_saved += 1
                        last_cx, last_cy = cx, cy
                        if seg_saved == 1:
                            print(f"  {vname[:20]} motion at frame {fi}", flush=True)

            cooldown = int(fps * 0.3)
        else:
            if cooldown > 0:
                cooldown -= 1
            elif seg_saved > 0:
                print(f"  -> {seg_saved} frames", flush=True)
                seg_saved = 0
                last_cx, last_cy = None, None

    if seg_saved > 0:
        print(f"  -> {seg_saved} frames", flush=True)
    cap.release()

after = len(list(OUTPUT.glob("*.jpg")))
print(f"\nDone! +{after-before} new frames. Total: {after}")
