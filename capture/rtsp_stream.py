import logging
from queue import Queue
from threading import Event, Thread

import cv2

log = logging.getLogger(__name__)


class RTSPStream:
    def __init__(self, url, reconnect_interval=5, timeout=10, max_queue=30, decode_every_n=5):
        self.url = url
        self.reconnect_interval = reconnect_interval
        self.timeout = timeout
        self.decode_every_n = decode_every_n
        self._cap = None
        self._queue = Queue(maxsize=max_queue)
        self._stop = Event()
        self._thread = None
        self._frame_count = 0

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

    def _connect(self):
        """Create and open a VideoCapture with timeouts applied."""
        cap = cv2.VideoCapture()
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.timeout > 0:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout * 1000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout * 1000)
        cap.open(self.url)
        return cap

    def _reader(self):
        while not self._stop.is_set():
            if self._cap is None:
                self._cap = self._connect()
                if not self._cap.isOpened():
                    log.warning("Failed to open stream, retrying in %ds", self.reconnect_interval)
                    self._cap.release()
                    self._cap = None
                    self._stop.wait(self.reconnect_interval)
                    continue
                log.info("Connected to RTSP stream")

            # grab() advances the buffer without decoding — cheap
            if not self._cap.grab():
                log.warning("Lost frame, reconnecting...")
                self._cap.release()
                self._cap = None
                self._stop.wait(self.reconnect_interval)
                continue

            self._frame_count += 1
            if self.decode_every_n > 1 and self._frame_count % self.decode_every_n != 0:
                continue  # grabbed but not decoded — skip

            ret, frame = self._cap.retrieve()
            if not ret or frame is None:
                log.warning("Failed to decode frame, skipping...")
                continue

            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Exception:
                    pass
            self._queue.put(frame)

        if self._cap:
            self._cap.release()
