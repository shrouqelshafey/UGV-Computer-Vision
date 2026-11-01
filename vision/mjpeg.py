"""Read complete HTTP MJPEG parts and skip isolated damaged JPEG images."""

from http.client import HTTPException
import urllib.error
import urllib.request

import cv2
import numpy as np

MAX_FRAME_BYTES = 512 * 1024
MAX_LINE_BYTES = 1024
MAX_HEADER_BYTES = 4096
MAX_BAD_FRAMES = 5


class _StreamError(ValueError):
    """A framing error with a safe message containing no URL or credentials."""


def _valid_jpeg_headers(image):
    """Check complete markers and DQT bounds before handing data to OpenCV."""
    if len(image) < 4 or not image.startswith(b"\xff\xd8") or not image.endswith(b"\xff\xd9"):
        return False
    position = 2
    while position < len(image) - 2:
        if image[position] != 0xFF:
            return False
        while position < len(image) and image[position] == 0xFF:
            position += 1
        if position >= len(image):
            return False
        marker = image[position]
        position += 1
        if marker in (0x00, 0xD8, 0xD9):
            return False
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if position + 2 > len(image) - 2:
            return False
        length = int.from_bytes(image[position:position + 2], "big")
        end = position + length
        if length < 2 or end > len(image) - 2:
            return False
        if marker == 0xDB:
            # A table needs its descriptor and 64 one-byte or two-byte entries.
            table_position = position + 2
            if table_position == end:
                return False
            while table_position < end:
                descriptor = image[table_position]
                precision, table_id = descriptor >> 4, descriptor & 0x0F
                if precision not in (0, 1) or table_id > 3:
                    return False
                table_position += 1 + 64 * (precision + 1)
                if table_position > end:
                    return False
        if marker == 0xDA:
            # The decoder checks entropy data, which uses different marker rules.
            return True
        position = end
    return False


def _error_message(error):
    if isinstance(error, _StreamError):
        return str(error)
    if isinstance(error, TimeoutError):
        return "Camera HTTP stream timed out."
    if isinstance(error, urllib.error.HTTPError):
        return f"Camera returned HTTP status {error.code}."
    if isinstance(error, urllib.error.URLError):
        if isinstance(error.reason, TimeoutError):
            return "Camera HTTP connection timed out."
        return "Could not connect to the camera HTTP stream."
    return f"Camera HTTP stream failed ({type(error).__name__})."


class MJPEGStream:
    """VideoCapture-like transport; the caller owns threading and reconnection."""

    def __init__(self, url, timeout=3.0):
        self.stream = None
        self._opened = False
        self._boundary = None
        self._consecutive_bad = 0
        self.last_error = None
        self.skipped_frames = 0
        try:
            request = urllib.request.Request(url, headers={"Connection": "close"})
            self.stream = urllib.request.urlopen(request, timeout=timeout)
            content_type = self.stream.headers.get_content_type().lower()
            boundary = self.stream.headers.get_param("boundary", header="Content-Type")
            if content_type != "multipart/x-mixed-replace":
                raise _StreamError("Camera response is not an MJPEG multipart stream.")
            if (not isinstance(boundary, str) or not 1 <= len(boundary) <= 70
                    or any(not 33 <= ord(char) <= 126 for char in boundary)):
                raise _StreamError("Invalid MJPEG multipart boundary.")
            self._boundary = b"--" + boundary.encode("ascii")
            self._opened = True
        except Exception as error:
            self._fail(error)
            print(f"[WARN] {self.last_error}")

    def isOpened(self):
        return self._opened

    def _fail(self, error):
        self.last_error = _error_message(error)
        self.release()

    def _readline(self):
        line = self.stream.readline(MAX_LINE_BYTES + 1)
        if not line:
            raise _StreamError("Camera HTTP stream ended while reading a part header.")
        if len(line) > MAX_LINE_BYTES or not line.endswith(b"\n"):
            raise _StreamError("MJPEG header line is too long or incomplete.")
        return line

    def _read_part(self):
        # HTTPResponse removes HTTP chunk framing. These lines are MIME framing.
        consumed = 0
        while True:
            line = self._readline()
            consumed += len(line)
            if consumed > MAX_HEADER_BYTES:
                raise _StreamError("MJPEG boundary preamble is too long.")
            value = line.rstrip(b"\r\n")
            if value == self._boundary:
                break
            if value == self._boundary + b"--":
                raise _StreamError("Camera HTTP stream ended.")
            if value:
                raise _StreamError("Invalid MJPEG part boundary.")

        headers = {}
        consumed = 0
        while True:
            line = self._readline()
            consumed += len(line)
            if consumed > MAX_HEADER_BYTES:
                raise _StreamError("MJPEG part headers are too large.")
            if line in (b"\r\n", b"\n"):
                break
            name, separator, value = line.partition(b":")
            name = name.strip().lower()
            if not separator or not name or name in headers:
                raise _StreamError("Invalid or repeated MJPEG part header.")
            headers[name] = value.strip()

        length = headers.get(b"content-length", b"")
        if not length.isdigit() or not 1 <= int(length) <= MAX_FRAME_BYTES:
            raise _StreamError("Invalid or oversized MJPEG Content-Length.")
        content_type = headers.get(b"content-type", b"image/jpeg").split(b";", 1)[0].lower()
        if content_type != b"image/jpeg":
            raise _StreamError("MJPEG part is not a JPEG image.")

        remaining = int(length)
        image = bytearray()
        while remaining:
            # Read exactly this image, never the next part's boundary or headers.
            chunk = self.stream.read(remaining)
            if not chunk:
                raise _StreamError("Camera HTTP stream ended during a JPEG image.")
            if len(chunk) > remaining:
                raise _StreamError("MJPEG image exceeded its declared length.")
            image.extend(chunk)
            remaining -= len(chunk)
        return bytes(image)

    def read(self):
        if not self._opened:
            return False, None
        try:
            while self._opened:
                image = self._read_part()
                frame = None
                if _valid_jpeg_headers(image):
                    try:
                        frame = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
                    except cv2.error:
                        pass
                if frame is not None:
                    self._consecutive_bad = 0
                    return True, frame
                # One bad JPEG should not force an otherwise live reconnect.
                self.skipped_frames += 1
                self._consecutive_bad += 1
                if self._consecutive_bad >= MAX_BAD_FRAMES:
                    raise _StreamError("Five consecutive camera JPEG images were invalid.")
        except (OSError, HTTPException, _StreamError) as error:
            self._fail(error)
        return False, None

    def release(self):
        self._opened = False
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
