#!/usr/bin/env python3
"""
Video watcher: monitors /mnt/ftp for new MP4 files, runs YOLO inference,
and copies videos with squirrel detections (confidence >= 0.5) to /mnt/datacollected.

Runs as a systemd service on the datacollector server.
"""
import json
import shutil
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# --- Config ---
WATCH_DIR = Path("/mnt/ftp")
OUTPUT_DIR = Path("/mnt/datacollected")
STATE_FILE = Path("/opt/squirrel-video-watcher/processed.json")
MODEL_PATH = Path("/opt/squirrel-video-watcher/models/scratch-v1.pt")
CONF_THRESHOLD = 0.5
POLL_INTERVAL = 10  # seconds
FRAME_SKIP = 15      # process every 15th frame (~1.6 fps on 25fps video)
MAX_W = 960         # resize width for faster inference


def log(msg: str):
    """Print with flush for systemd journal."""
    print(msg, flush=True)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
log(f"[WATCHER] Device: {device}")
if device == "cpu":
    log("[WATCHER] WARNING: Running on CPU -- this will be slow!")

# --- Load model ---
log(f"[WATCHER] Loading model from {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
log("[WATCHER] Model loaded")


def load_state() -> set:
    """Load set of already-processed filenames."""
    if STATE_FILE.exists():
        return set(json.loads(STATE_FILE.read_text()))
    return set()


def save_state(state: set):
    """Persist processed filenames."""
    STATE_FILE.write_text(json.dumps(sorted(state), indent=2))


def video_has_squirrel(path: Path) -> bool:
    """Run YOLO on video frames. Return True if any detection >= CONF_THRESHOLD."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        log(f"  [SKIP] Cannot open video: {path.name}")
        return False

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f"  Scanning {path.name} ({total_frames} frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        # Resize to speed up on GT 1030
        h, w = frame.shape[:2]
        if w > MAX_W:
            scale = MAX_W / w
            frame = cv2.resize(frame, (MAX_W, int(h * scale)))

        # Run inference
        results = model(frame, verbose=False, device=device)

        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf >= CONF_THRESHOLD:
                    cap.release()
                    log(f"  [HIT] Frame {frame_count}: confidence {conf:.3f}")
                    return True

    cap.release()
    return False


def main():
    log(f"[WATCHER] Watching {WATCH_DIR} for new MP4 files")
    log(f"[WATCHER] Polling every {POLL_INTERVAL}s, CONF_THRESHOLD={CONF_THRESHOLD}")
    log(f"[WATCHER] Output: {OUTPUT_DIR}")

    processed = load_state()
    log(f"[WATCHER] Already processed: {len(processed)} videos")

    while True:
        try:
            videos = sorted(WATCH_DIR.glob("*.mp4"))
            new_videos = [v for v in videos if v.name not in processed]

            for v in new_videos:
                log(f"\n[NEW] {v.name} ({v.stat().st_size / 1e6:.0f} MB)")
                try:
                    found = video_has_squirrel(v)
                except Exception as e:
                    log(f"  [ERR] {e}")
                    found = False

                if found:
                    dest = OUTPUT_DIR / v.name
                    shutil.copy2(str(v), str(dest))
                    log(f"  [COPY] -> {dest}")
                else:
                    log(f"  [NO] No squirrel with confidence >= {CONF_THRESHOLD}")

                processed.add(v.name)
                save_state(processed)

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            log("\n[WATCHER] Shutting down")
            save_state(processed)
            break
        except Exception as e:
            log(f"[WATCHER] Error: {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
