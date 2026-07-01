"""
Extract distinctive frames from Reolink videos.
- Samples at 3 FPS
- Deduplicates near-identical frames via MSE comparison
- Output: H:\reolink\training_frames\ (flat)

Usage:
    python extract_training_frames.py
    python extract_training_frames.py --fps 3 --mse 200
"""
import cv2
import numpy as np
import argparse
from pathlib import Path

REOLINK_DIR = Path("H:/reolink")
OUTPUT_DIR = REOLINK_DIR / "training_frames"

EXTRACT_FPS = 3
MSE_THRESHOLD = 200  # lower = more frames (more sensitive), higher = fewer


def extract_frames(video_path, output_dir, extract_fps, mse_thresh):
    """Sample video at extract_fps, save frames that differ enough from last."""
    video_name = video_path.stem
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"  [ERR] Cannot open {video_name}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = total_frames / video_fps
    print(f"  {total_frames} frames, {video_fps:.1f} FPS, {video_duration:.0f}s, {width}x{height}")

    frame_interval = max(1, int(round(video_fps / extract_fps)))
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    skipped_similar = 0
    frame_idx = 0
    last_thumb = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            thumb = cv2.resize(frame, (320, 180))

            # Check if distinctive vs last saved
            is_distinctive = True
            if last_thumb is not None:
                diff = cv2.absdiff(thumb.astype(np.int16), last_thumb.astype(np.int16))
                mse = np.mean(diff ** 2)
                if mse < mse_thresh:
                    is_distinctive = False
                    skipped_similar += 1

            if is_distinctive:
                ts = frame_idx / video_fps
                out_name = f"{video_name}_{frame_idx:06d}_{ts:.1f}s.jpg"
                cv2.imwrite(str(output_dir / out_name), frame)
                saved_count += 1
                last_thumb = thumb.copy()

        frame_idx += 1

    cap.release()
    if skipped_similar > 0:
        print(f"  Saved {saved_count} frames ({skipped_similar} skipped as too similar)")
    else:
        print(f"  Saved {saved_count} frames")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Extract distinctive training frames from videos")
    parser.add_argument("--fps", type=float, default=EXTRACT_FPS,
                        help=f"Frame sampling rate (default: {EXTRACT_FPS})")
    parser.add_argument("--mse", type=float, default=MSE_THRESHOLD,
                        help=f"Min pixel change to be distinctive, 320x180 thumb (default: {MSE_THRESHOLD})")
    args = parser.parse_args()

    videos = sorted(REOLINK_DIR.glob("*.mp4"))
    if not videos:
        print(f"No .mp4 files found in {REOLINK_DIR}")
        return

    print(f"Found {len(videos)} videos")
    print(f"Sampling at {args.fps} FPS, MSE threshold: {args.mse}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    total = 0
    for i, v in enumerate(videos):
        print(f"[{i+1}/{len(videos)}] {v.name}")
        frames = extract_frames(v, OUTPUT_DIR, args.fps, args.mse)
        total += frames
        print()

    print("=" * 60)
    print(f"Done! {total} total frames -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
