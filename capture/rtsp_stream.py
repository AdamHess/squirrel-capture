import logging
from queue import Queue
from threading import Event, Thread

import cv2

log = logging.getLogger(__name__)


class RTSPStream:
    def __init__(self, url, reconnect_interval=5, timeout=10, max_queue=30):
        self.url = url
        self.reconnect_interval = reconnect_interval
        self.timeout = timeout
        self._cap = None
        self._queue = Queue(maxsize=max_queue)
        self._stop = Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = Thread(target=self._reader, daemon=True)
        self._thread.start()
        log.info("RTSP reader started")
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        if self._cap:
            self._cap.release()
        log.info("RTSP reader stopped")

    def read(self):
        try:
            return self._queue.get_nowait()
        except Exception:
            return None

    @property
    def is_active(self):
        return not self._queue.empty() or (self._thread and self._thread.is_alive())

    def _reader(self):
        while not self._stop.is_set():
            if self._cap is None:
                self._cap = cv2.VideoCapture(self.url)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not self._cap.isOpened():
                    log.warning("Failed to open stream, retrying in %ds", self.reconnect_interval)
                    self._cap = None
                    self._stop.wait(self.reconnect_interval)
                    continue
                log.info("Connected to RTSP stream")

            ret, frame = self._cap.read()
            if not ret or frame is None:
                log.warning("Lost frame, reconnecting...")
                self._cap.release()
                self._cap = None
                self._stop.wait(self.reconnect_interval)
                continue

            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Exception:
                    pass
            self._queue.put(frame)

        if self._cap:
            self._cap.release()
