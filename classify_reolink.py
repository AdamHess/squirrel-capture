"""
Classify H:\reolink videos using scratch-v1.
- Sort videos by timestamp (from filename)
- Sample at 2 FPS, check for squirrels
- Keep: videos WITH squirrels + the previous and next video in sorted order
- Delete: everything else

Usage:
    python classify_reolink.py
    python classify_reolink.py --conf 0.25 --dry-run
"""
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO

REOLINK_DIR = Path("H:/reolink")
OUTPUT_DIR = Path("D:/squrrel_training/reolink_extracted")
MODEL_PATH = "D:/squrrel_training/squirrel-capture/deploy/scratch-v1.pt"
CONF_THRESHOLD = 0.25
FPS_SAMPLE = 2


def has_squirrel(video_path, model, conf_threshold):
    """Returns True if any squirrel detected in video at 2 FPS."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30

    frame_interval = max(1, int(round(video_fps / FPS_SAMPLE)))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            results = model(frame, imgsz=960, conf=conf_threshold, verbose=False)
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                cap.release()
                return True

        frame_idx += 1

    cap.release()
    return False


def main():
    parser = argparse.ArgumentParser(description="Classify Reolink videos by squirrel content")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                        help=f"Confidence threshold (default: {CONF_THRESHOLD})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't delete anything, just report")
    args = parser.parse_args()

    videos = sorted(REOLINK_DIR.glob("*.mp4"))
    if not videos:
        print(f"No .mp4 files found in {REOLINK_DIR}")
        return

    total_size_gb = sum(v.stat().st_size for v in videos) / (1024**3)
    print(f"Found {len(videos)} videos in time-sorted order ({total_size_gb:.1f} GB total)")
    print(f"Confidence threshold: {args.conf}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Step 1: load model once
    model = YOLO(str(MODEL_PATH))

    # Step 2: classify each video
    has_squirrel_flags = []
    for i, v in enumerate(videos):
        size_mb = v.stat().st_size / (1024**2)
        print(f"[{i+1}/{len(videos)}] Scanning {v.name}  ({size_mb:.0f} MB) ...", end=" ", flush=True)
        result = has_squirrel(v, model, args.conf)
        has_squirrel_flags.append(result)
        print("HAS SQUIRREL" if result else "no squirrel")

    # Step 3: determine which to keep (squirrel video + prev + next in sorted order)
    keep_indices = set()
    for i, has_it in enumerate(has_squirrel_flags):
        if has_it:
            keep_indices.add(i)
            if i > 0:
                keep_indices.add(i - 1)  # previous video
            if i < len(videos) - 1:
                keep_indices.add(i + 1)  # next video

    # Step 4: keep or delete
    print("\n" + "=" * 60)
    kept = 0
    deleted = 0
    kept_size = 0
    deleted_size = 0

    for i, v in enumerate(videos):
        if i in keep_indices:
            kept += 1
            kept_size += v.stat().st_size
            reason = "squirrel" if has_squirrel_flags[i] else "adjacent to squirrel video"
            print(f"  ✓ KEEP  [{i+1}] {v.name}  ({reason})")
        else:
            deleted += 1
            deleted_size += v.stat().st_size
            print(f"  ✗ DEL   [{i+1}] {v.name}")
            if not args.dry_run:
                v.unlink()

    print("\n" + "=" * 60)
    print(f"SUMMARY:")
    print(f"  Videos with squirrels:   {sum(has_squirrel_flags)}")
    print(f"  Kept (squirrel ±1):      {kept} videos ({kept_size/(1024**3):.1f} GB)")
    print(f"  Deleted (no squirrel):   {deleted} videos ({deleted_size/(1024**3):.1f} GB freed)")
    print(f"  Remaining on disk:       {kept} videos")


if __name__ == "__main__":
    main()
