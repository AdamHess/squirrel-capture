import logging

from ultralytics import YOLO

log = logging.getLogger(__name__)

COCO_SQUIRREL_ID = 9


def _resolve(path_or_name: str) -> str:
    from models.registry import resolve_model_path

    return resolve_model_path(path_or_name)


class YOLODetector:
    def __init__(
        self,
        model_path="nyc-backyard-v1",
        conf_threshold=0.25,
        iou_threshold=0.45,
        target_classes=None,
        device="cpu",
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or []
        self.device = device
        resolved = _resolve(model_path)
        log.info("Loading model %s (resolved: %s) on %s ...", model_path, resolved, device)
        self.model = YOLO(resolved)
        log.info("Model loaded")

    def detect(self, frame):
        import time
        t0 = time.time()
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )[0]
        inference_ms = (time.time() - t0) * 1000

        detections = []
        if results.boxes is None:
            return detections

        for box, cls_id, conf in zip(
            results.boxes.xyxy, results.boxes.cls, results.boxes.conf, strict=True
        ):
            cls_id = int(cls_id)
            conf = float(conf)

            if self.target_classes and cls_id not in self.target_classes:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    "class_name": results.names[cls_id],
                    "confidence": conf,
                    "inference_ms": round(inference_ms, 1),
                }
            )

        return detections

    def detect_with_visualization(self, frame):
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )[0]

        annotated = results.plot()
        detections = self.parse_results(results)

        return annotated, detections

    @staticmethod
    def parse_results(results):
        detections = []
        if results.boxes is None:
            return detections

        for box, cls_id, conf in zip(
            results.boxes.xyxy, results.boxes.cls, results.boxes.conf, strict=True
        ):
            cls_id = int(cls_id)
            conf = float(conf)
            x1, y1, x2, y2 = map(int, box.tolist())
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    "class_name": results.names[cls_id],
                    "confidence": conf,
                }
            )

        return detections
