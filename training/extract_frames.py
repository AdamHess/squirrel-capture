"""Extract distinct frames from a video for hand labeling."""
import cv2
import os
import sys
from pathlib import Path

video = r"C:\Users\Adam\Downloads\reolink\Backyard-0-20260626142157-20260626142657.mp4"
output = Path(r"C:\Users\Adam\squirrel-capture\data\exports\hand-label-0626\images")
output.mkdir(parents=True, exist_ok=True)

segments = [
    (29, 35),
    (45, None),  # None = to end
]

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps
print(f"Video: {fps:.1f} fps, {total_frames} frames, {duration:.0f}s")

saved = 0
for start_s, end_s in segments:
    start_f = int(start_s * fps)
    end_f = int(end_s * fps) if end_s else total_frames
    print(f"\nSegment {start_s}s-{end_s or 'end'}s: frames {start_f}-{end_f}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frame_idx = start_f
    last_gray = None

    while frame_idx < end_f:
        ret, frame = cap.read()
        if not ret:
            break

        # Dedup: skip frames too similar to the last saved one
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if last_gray is not None:
            diff = cv2.absdiff(gray, last_gray).mean()
            if diff < 15:  # skip near-duplicate frames
                frame_idx += 1
                continue

        ts = int(video.rsplit('-', 1)[-1].replace('.mp4', ''))
        fname = f"sq_{ts}_{frame_idx:06d}.jpg"
        cv2.imwrite(str(output / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
        last_gray = gray
        print(f"  Saved frame {frame_idx} ({diff:.1f} diff)" if saved > 1 else f"  Saved frame {frame_idx} (first)")
        frame_idx += 1

cap.release()
print(f"\nDone! {saved} frames saved to {output}")
