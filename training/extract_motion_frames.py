"""Extract distinct frames via MOG2 motion detection. No ML model used."""
import cv2
import numpy as np
import os
from pathlib import Path

VIDEO_DIR = r"C:\Users\Adam\Desktop\Backyard_2026-06-27_0730-0930"
OUTPUT = Path(r"C:\Users\Adam\squirrel-capture\data\exports\motion-extract-0627\images")
OUTPUT.mkdir(parents=True, exist_ok=True)

MOTION_MIN_AREA = 300    # min contour px at 640 scale
MOG2_VAR = 36
DEDUP_DIST = 30          # centroid must move this many px at 640 scale
PROC_W = 640

videos = sorted([os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])
print(f"Videos: {len(videos)}")

total_saved = 0

for vi, vpath in enumerate(videos):
    vname = os.path.basename(vpath)
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = PROC_W / W
    print(f"\n[{vi+1}/{len(videos)}] {vname}  {W}x{H} {fps:.1f}fps {total_f}f")

    mog2 = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=MOG2_VAR, detectShadows=True)
    last_cx, last_cy = None, None
    cooldown = 0
    seg_saved = 0
    fi = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1

        # Scale down for MOG2 processing
        small = cv2.resize(frame, (PROC_W, int(H * scale)))
        fg = mog2.apply(small)
        fg = cv2.medianBlur(fg, 5)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in contours if cv2.contourArea(c) > MOTION_MIN_AREA]

        if big:
            # Get centroid of largest motion blob
            largest = max(big, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                if last_cx is not None:
                    dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                else:
                    dist = DEDUP_DIST + 1

                if dist > DEDUP_DIST:
                    # Save full-res frame
                    fname = f"me_{vi+1:02d}_{fi:06d}.jpg"
                    outpath = OUTPUT / fname
                    if not outpath.exists():
                        cv2.imwrite(str(outpath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        seg_saved += 1
                        last_cx, last_cy = cx, cy
                        if seg_saved == 1:
                            print(f"    Motion at frame {fi}", end="")

            cooldown = int(fps * 0.5)
        else:
            if cooldown > 0:
                cooldown -= 1
            elif seg_saved > 0:
                print(f"\r    [{vi+1}] {seg_saved} frames")
                total_saved += seg_saved
                seg_saved = 0
                last_cx, last_cy = None, None

    # End of video
    if seg_saved > 0:
        print(f"\r    [{vi+1}] {seg_saved} frames")
        total_saved += seg_saved
    cap.release()

print(f"\n\nDone! {total_saved} frames -> {OUTPUT}")
