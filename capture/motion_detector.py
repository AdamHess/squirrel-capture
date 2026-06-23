import time
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


class MotionDetector:
    def __init__(self, method="mog2", min_area=3000, threshold=25, cooldown=1.0):
        self.method = method
        self.min_area = min_area
        self.threshold = threshold
        self.cooldown = cooldown
        self._last_trigger = 0.0
        self._bg_subtractor = None
        self._prev_frame = None
        self._setup()

    def _setup(self):
        if self.method == "mog2":
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=self.threshold, detectShadows=True
            )
        elif self.method == "frame_diff":
            self._prev_frame = None
        elif self.method == "none":
            pass
        else:
            log.warning("Unknown motion method '%s', falling back to 'none'", self.method)
            self.method = "none"

    def detect(self, frame):
        if self.method == "none":
            return False

        now = time.time()
        if now - self._last_trigger < self.cooldown:
            return False

        motion = False

        if self.method == "mog2":
            fg_mask = self._bg_subtractor.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) >= self.min_area:
                    motion = True
                    break

        elif self.method == "frame_diff":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if self._prev_frame is None:
                self._prev_frame = gray
                return False
            diff = cv2.absdiff(self._prev_frame, gray)
            _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) >= self.min_area:
                    motion = True
                    break
            self._prev_frame = gray

        if motion:
            self._last_trigger = now
            return True

        return False

    def reset(self):
        self._setup()
        self._last_trigger = 0.0
        log.info("Motion detector reset")
