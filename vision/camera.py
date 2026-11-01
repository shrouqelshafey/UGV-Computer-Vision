"""Read continuously and keep only the latest available camera frame."""

import threading

from vision.mjpeg import MJPEGStream


class LatestFrameCamera:
    """Separate stream reading from detection so old frames do not accumulate."""

    def __init__(self, url):
        # Parse complete JPEG parts directly, so one damaged image does not
        # terminate an otherwise usable stream inside the FFmpeg decoder.
        self._capture = MJPEGStream(url, timeout=8.0)
        self._opened = self._capture.isOpened()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._pending = None
        self._pending_skipped = 0
        self._closed = False
        self._worker = None
        self._error = None
        self._ended = False
        self.dropped_frames = 0
        self.skipped_frames = 0
        self.last_error = None
        if self._opened:
            self._worker = threading.Thread(
                target=self._receive_frames, name="camera-reader", daemon=True
            )
            self._worker.start()
        else:
            self._capture.release()

    def isOpened(self):
        return self._opened and not self._closed and not self._ended

    def _receive_frames(self):
        try:
            while not self._stop.is_set():
                result = self._capture.read()
                if not result[0]:
                    # EOF belongs to this connection. Let main open a new one.
                    reason = getattr(self._capture, "last_error", None)
                    if isinstance(reason, str):
                        self.last_error = reason
                    break
                with self._condition:
                    if self._stop.is_set():
                        break
                    if self._pending is not None and self._pending[0]:
                        self.dropped_frames += 1
                    # Replace an unprocessed frame instead of extending a queue.
                    self._pending = result
                    skipped = getattr(self._capture, "skipped_frames", 0)
                    self._pending_skipped = skipped if isinstance(skipped, int) else 0
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._error = error
        finally:
            # Only the reader releases the backend: never release during read().
            self._capture.release()
            with self._condition:
                # Drop stale frames and notify consumers only after socket close.
                self._pending = None
                self._ended = True
                self._condition.notify_all()

    def read(self):
        with self._condition:
            while (self._pending is None and not self._closed
                   and self._opened and not self._ended):
                self._condition.wait()
            if self._error is not None and not self._closed:
                raise self._error
            if self._closed or not self._opened or self._ended:
                return False, None
            result = self._pending
            self.skipped_frames = self._pending_skipped
            # A delivered frame is consumed, so it cannot confirm four times.
            self._pending = None
            return result

    def release(self):
        with self._condition:
            if self._closed:
                return self._worker is None or not self._worker.is_alive()
            self._closed = True
            self._stop.set()
            self._condition.notify_all()
        if self._worker is not None:
            # Allow the 8-second socket timeout to finish before a replacement opens.
            self._worker.join(timeout=9.0)
            return not self._worker.is_alive()
        return True


def open_camera(url):
    return LatestFrameCamera(url)
