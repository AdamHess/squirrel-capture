import logging
import time
from pathlib import Path

import cv2

log = logging.getLogger(__name__)


class AutoLabeler:
    def __init__(
        self,
        output_dir="data",
        label_format="yolo",
        save_raw=True,
        save_labeled=True,
        save_annotated=False,
        min_confidence=0.3,
        track_cooldown=10,
        quality=None,
    ):
        self.output_dir = Path(output_dir)
        self.label_format = label_format
        self.save_raw = save_raw
        self.save_labeled = save_labeled
        self.save_annotated = save_annotated
        self.min_confidence = min_confidence
        self.track_cooldown = track_cooldown
        self.quality = quality or {
            "min_blur": 50,
            "min_box_area": 2000,
            "max_edge_margin": 10,
        }
        self._last_track_save: dict[int, float] = {}
        self._saved_count = 0
        self._raw_count = 0
        self._labeled_count = 0

        self.raw_dir = self.output_dir / "raw"
        self.labeled_dir = self.output_dir / "labeled"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.labeled_dir.mkdir(parents=True, exist_ok=True)
        (self.labeled_dir / "images").mkdir(exist_ok=True)
        (self.labeled_dir / "labels").mkdir(exist_ok=True)

    def _filter_quality(self, frame, detections):
        """Remove detections that fail quality checks. Returns filtered list or None if frame is junk."""
        q = self.quality

        # Blur check — if the whole frame is blurry, nothing is worth saving
        if q.get("min_blur", 0) > 0:
            var = cv2.Laplacian(frame, cv2.CV_64F).var()
            if var < q["min_blur"]:
                log.debug("Skipped blurry frame (Laplacian var=%.1f)", var)
                return None

        kept = []
        h, w = frame.shape[:2]
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]

            # Edge check — reject detections clipped at frame boundary
            margin = q.get("max_edge_margin", 0)
            if margin > 0 and (x1 < margin or y1 < margin or x2 > w - margin or y2 > h - margin):
                log.debug("Rejected edge detection [%d,%d,%d,%d]", x1, y1, x2, y2)
                continue

            # Size check — reject detections too small to be useful
            min_area = q.get("min_box_area", 0)
            if min_area > 0 and (x2 - x1) * (y2 - y1) < min_area:
                log.debug("Rejected small detection (%d px)", (x2 - x1) * (y2 - y1))
                continue

            kept.append(d)

        return kept if kept else None

    def _should_save(self, detections):
        """Skip if every tracked detection was saved within cooldown."""
        now = time.time()
        any_new = False
        for d in detections:
            tid = d.get("track_id")
            if tid is None:
                return True  # untracked detection, always save
            last = self._last_track_save.get(tid, 0)
            if now - last >= self.track_cooldown:
                any_new = True
        return any_new

    def save(self, frame, detections, timestamp=None):
        kept = [d for d in detections if d["confidence"] >= self.min_confidence]
        if not kept:
            return None

        kept = self._filter_quality(frame, kept)
        if not kept:
            return None

        if not self._should_save(kept):
            return None

        ts = timestamp or int(time.time() * 1000)
        filename = f"squirrel_{ts}"

        # Record save time for each tracked object
        now = time.time()
        for d in kept:
            if "track_id" in d:
                self._last_track_save[d["track_id"]] = now

        if self.save_raw:
            raw_path = str(self.raw_dir / f"{filename}.jpg")
            cv2.imwrite(raw_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            self._raw_count += 1

        if self.save_labeled:
            img_path = str(self.labeled_dir / "images" / f"{filename}.jpg")
            cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            self._write_label(filename, kept, frame.shape)
            self._labeled_count += 1

            if self.save_annotated:
                annotated = frame.copy()
                for d in kept:
                    x1, y1, x2, y2 = d["bbox"]
                    label = f"{d['class_name']} {d['confidence']:.2f}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
                    if "track_id" in d:
                        tid_label = f"ID: {d['track_id']}"
                        cv2.putText(
                            annotated,
                            tid_label,
                            (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 0),
                            1,
                        )
                anno_path = str(self.labeled_dir / "images" / f"{filename}_annotated.jpg")
                cv2.imwrite(anno_path, annotated)

        self._saved_count += 1
        log.debug("Saved %s (%d detections)", filename, len(kept))
        return filename

    def _write_label(self, filename, detections, img_shape):
        h, w = img_shape[:2]
        lines = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{d['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_path = self.labeled_dir / "labels" / f"{filename}.txt"
        label_path.write_text("\n".join(lines))

    @property
    def stats(self):
        return {
            "total_saved": self._saved_count,
            "raw_count": self._raw_count,
            "labeled_count": self._labeled_count,
        }
