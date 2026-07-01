import logging
from collections import OrderedDict

log = logging.getLogger(__name__)


class DetectionTracker:
    def __init__(self, method="bytetrack", max_lost=30, iou_threshold=0.3):
        self.method = method
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self._tracks = OrderedDict()
        self._next_id = 0

    def update(self, detections):
        if self.method == "iou":
            return self._iou_update(detections)
        elif self.method == "bytetrack":
            return self._bytetrack_update(detections)
        return detections

    def _bbox_center(self, bbox):
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _bbox_iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    def _iou_update(self, detections):
        matched = []
        for det in detections:
            best_id = None
            best_iou = 0
            for track_id, track in list(self._tracks.items()):
                iou = self._bbox_iou(track.bbox, det["bbox"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_id = track_id
            if best_id is not None:
                self._tracks[best_id].update(det, frame_id=0)
                det["track_id"] = best_id
                matched.append(det)
            else:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = TrackState(track_id, det)
                det["track_id"] = track_id
                matched.append(det)

        self._prune_tracks()
        return matched

    def _bytetrack_update(self, detections):
        matched = []
        unmatched_dets = []

        for det in detections:
            best_id = None
            best_iou = 0
            for track_id, track in list(self._tracks.items()):
                iou = self._bbox_iou(track.bbox, det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = track_id
            if best_id is not None and best_iou >= self.iou_threshold:
                self._tracks[best_id].update(det, frame_id=0)
                det["track_id"] = best_id
                det["frames_seen"] = self._tracks[best_id].frames_seen
                matched.append(det)
            else:
                unmatched_dets.append(det)

        for det in unmatched_dets:
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = TrackState(track_id, det)
            det["track_id"] = track_id
            det["frames_seen"] = 1
            matched.append(det)

        self._prune_tracks()
        return matched

    def _prune_tracks(self):
        dead = [tid for tid, t in self._tracks.items() if t.lost > self.max_lost]
        for tid in dead:
            del self._tracks[tid]

    def reset(self):
        self._tracks.clear()
        log.info("Tracker reset")


class TrackState:
    def __init__(self, track_id, detection):
        self.track_id = track_id
        self.bbox = detection["bbox"]
        self.class_id = detection["class_id"]
        self.class_name = detection["class_name"]
        self.confidence = detection["confidence"]
        self.lost = 0
        self.frames_seen = 1
        self.best_conf = detection["confidence"]

    def update(self, detection, frame_id=0):
        self.bbox = detection["bbox"]
        self.confidence = detection["confidence"]
        self.lost = 0
        self.frames_seen += 1
        if detection["confidence"] > self.best_conf:
            self.best_conf = detection["confidence"]

    @property
    def is_confirmed(self):
        return self.frames_seen >= 3

    @property
    def best_detection(self):
        return {
            "bbox": self.bbox,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.best_conf,
            "track_id": self.track_id,
        }
