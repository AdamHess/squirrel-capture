import logging

import cv2
import numpy as np
from ultralytics import YOLO

log = logging.getLogger(__name__)

COCO_SQUIRREL_ID = 9


class YOLODetector:
    def __init__(self, model_path="yolo11n.pt",
                 conf_threshold=0.25, iou_threshold=0.45,
                 target_classes=None, device="cpu"):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or []
        self.device = device
        log.info("Loading model %s on %s ...", model_path, device)
        self.model = YOLO(model_path)
        log.info("Model loaded")

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        if results.boxes is None:
            return detections

        for box, cls_id, conf in zip(results.boxes.xyxy,
                                     results.boxes.cls,
                                     results.boxes.conf):
            cls_id = int(cls_id)
            conf = float(conf)

            if self.target_classes and cls_id not in self.target_classes:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": cls_id,
                "class_name": results.names[cls_id],
                "confidence": conf,
            })

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

        for box, cls_id, conf in zip(results.boxes.xyxy,
                                     results.boxes.cls,
                                     results.boxes.conf):
            cls_id = int(cls_id)
            conf = float(conf)
            x1, y1, x2, y2 = map(int, box.tolist())
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": cls_id,
                "class_name": results.names[cls_id],
                "confidence": conf,
            })

        return detections
