import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import yaml

from capture.motion_detector import MotionDetector
from capture.rtsp_stream import RTSPStream
from detection.tracker import DetectionTracker
from detection.yolo_detector import YOLODetector
from labeling.auto_labeler import AutoLabeler

log = logging.getLogger("pipeline")


class SquirrelPipeline:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.running = False
        self._fps_counter = 0
        self._fps_time = time.time()
        self._display_fps = 0

        cam = cfg["camera"]
        self.stream = RTSPStream(
            url=cam["rtsp_url"],
            reconnect_interval=cam.get("reconnect_interval", 5),
            timeout=cam.get("timeout", 10),
            decode_every_n=cam.get("decode_every_n", 5),
        )

        mot = cfg["motion"]
        if mot.get("enabled", True):
            self.motion = MotionDetector(
                method=mot.get("method", "mog2"),
                min_area=mot.get("min_area", 3000),
                threshold=mot.get("threshold", 25),
                cooldown=mot.get("cooldown", 1.0),
            )
        else:
            self.motion = None

        det = cfg["detection"]
        self.detector = YOLODetector(
            model_path=det.get("model", "yolo11n.pt"),
            conf_threshold=det.get("conf_threshold", 0.25),
            iou_threshold=det.get("iou_threshold", 0.45),
            target_classes=det.get("target_classes", []),
            device=det.get("device", "cpu"),
        )
        self._sweep_every = det.get("yolo_sweep", 12)
        self._sweep_counter = 0

        trk = cfg["tracker"]
        if trk.get("enabled", True):
            self.tracker = DetectionTracker(
                method=trk.get("method", "bytetrack"),
                max_lost=trk.get("max_lost", 30),
                iou_threshold=trk.get("iou_threshold", 0.3),
            )
        else:
            self.tracker = None

        cap = cfg["capture"]
        self.labeler = AutoLabeler(
            output_dir=cap.get("output_dir", "data"),
            label_format=cap.get("label_format", "yolo"),
            save_raw=cap.get("save_raw", True),
            save_labeled=cap.get("save_labeled", True),
            save_annotated=cap.get("save_annotated", False),
            min_confidence=cap.get("min_confidence", 0.3),
            track_cooldown=cap.get("track_cooldown", 10),
            quality=cap.get("quality", None),
        )

    def start(self):
        self.running = True
        self.stream.start()
        log.info("Pipeline started")

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        try:
            self._loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.stream.stop()
        stats = self.labeler.stats
        log.info(
            "Pipeline stopped. Saved %d images. Raw: %d, Labeled: %d.",
            stats["total_saved"],
            stats["raw_count"],
            stats["labeled_count"],
        )

    def _handle_signal(self, signum, frame):
        log.info("Received signal %d, shutting down...", signum)
        self.running = False

    def _loop(self):
        log.info("Entering main loop")
        warmup = True
        warmup_frames = 30
        frame_count = 0

        while self.running:
            frame = self.stream.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1

            if warmup:
                # Drain initial frames to let the stream stabilize.
                # MOG2 starts fresh on the first real detection frame below.
                if frame_count >= warmup_frames:
                    warmup = False
                    log.info("Warmup complete")
                continue

            self._update_fps()

            motion_detected = False
            if self.motion:
                motion_detected = self.motion.detect(frame)

            run_yolo = False
            if motion_detected:
                run_yolo = True
            elif self._sweep_every > 0:
                self._sweep_counter += 1
                if self._sweep_counter >= self._sweep_every:
                    self._sweep_counter = 0
                    run_yolo = True

            if not run_yolo:
                continue

            detections = self.detector.detect(frame)

            if not detections:
                continue

            if self.tracker:
                detections = self.tracker.update(detections)

            self.labeler.save(frame, detections)

            if self._display_fps > 0 and frame_count % 30 == 0:
                log.debug(
                    "FPS: %d, detections: %d, total saved: %d",
                    self._display_fps,
                    len(detections),
                    self.labeler.stats["total_saved"],
                )

    def _update_fps(self):
        self._fps_counter += 1
        elapsed = time.time() - self._fps_time
        if elapsed >= 1.0:
            self._display_fps = int(self._fps_counter / elapsed)
            self._fps_counter = 0
            self._fps_time = time.time()


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="Squirrel capture pipeline with auto-labeling")
    parser.add_argument(
        "-c", "--config", default="config.yaml", help="Path to config file (default: config.yaml)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    if not Path(args.config).exists():
        log.error("Config file not found: %s", args.config)
        sys.exit(1)

    pipeline = SquirrelPipeline(args.config)
    pipeline.start()


if __name__ == "__main__":
    main()
