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
        min_confidence=0.3,
        max_per_hour=120,
    ):
        self.output_dir = Path(output_dir)
        self.label_format = label_format
        self.save_raw = save_raw
        self.save_labeled = save_labeled
        self.min_confidence = min_confidence
        self.max_per_hour = max_per_hour
        self._hourly_count = 0
        self._hour_start = time.time()
        self._saved_count = 0

        self.raw_dir = self.output_dir / "raw"
        self.labeled_dir = self.output_dir / "labeled"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.labeled_dir.mkdir(parents=True, exist_ok=True)
        (self.labeled_dir / "images").mkdir(exist_ok=True)
        (self.labeled_dir / "labels").mkdir(exist_ok=True)

    def _check_rate_limit(self):
        now = time.time()
        if now - self._hour_start > 3600:
            self._hourly_count = 0
            self._hour_start = now
        if self._hourly_count >= self.max_per_hour:
            return False
        return True

    def save(self, frame, detections, timestamp=None):
        if not self._check_rate_limit():
            return None

        ts = timestamp or int(time.time() * 1000)
        filename = f"squirrel_{ts}"

        kept = [d for d in detections if d["confidence"] >= self.min_confidence]
        if not kept:
            return None

        if self.save_raw:
            raw_path = str(self.raw_dir / f"{filename}.jpg")
            cv2.imwrite(raw_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        if self.save_labeled:
            img_path = str(self.labeled_dir / "images" / f"{filename}.jpg")
            cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            self._write_label(filename, kept, frame.shape)

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

        self._hourly_count += 1
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
            "hourly_count": self._hourly_count,
            "raw_count": len(list(self.raw_dir.glob("*.jpg"))),
            "labeled_count": len(list((self.labeled_dir / "labels").glob("*.txt"))),
        }
