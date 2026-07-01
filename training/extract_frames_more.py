"""Re-extract frames with lower dedup threshold for gaps."""
import cv2
from pathlib import Path

video = r"C:\Users\Adam\Downloads\reolink\Backyard-0-20260626142157-20260626142657.mp4"
output = Path(r"C:\Users\Adam\squirrel-capture\data\exports\hand-label-0626\images")

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"FPS: {fps}, Total frames: {total_frames}")

# Full segment: 45s to end, but we need frames specifically:
# 1. Between 001402 and next (around frame 1402 to 1654 = ~234f gap)
# 2. After 001786 to end (frame 1786 to 1801)

# Let's just do 45s to end with a lower dedup threshold
start_f = 270  # 45s
end_f = total_frames

cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
saved = 0
last_gray = None
frame_idx = start_f

while frame_idx < end_f:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if last_gray is not None:
        diff = cv2.absdiff(gray, last_gray).mean()
        if diff < 8:  # lower threshold = more frames
            frame_idx += 1
            continue

    ts = int(video.rsplit('-', 1)[-1].replace('.mp4', ''))
    fname = f"sq_{ts}_{frame_idx:06d}.jpg"
    fpath = output / fname
    if fpath.exists():
        last_gray = gray  # still track for dedup against next
        frame_idx += 1
        continue
    cv2.imwrite(str(fpath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    saved += 1
    last_gray = gray
    if saved <= 5 or frame_idx > 1780:
        diff_str = f" (diff={diff:.1f})" if last_gray is not None and saved > 1 else ""
        print(f"  Frame {frame_idx} saved{diff_str}")
    frame_idx += 1

cap.release()
print(f"\nDone! {saved} new/overwritten frames")
print(f"Total in folder: {len(list(output.glob('*.jpg')))}")
