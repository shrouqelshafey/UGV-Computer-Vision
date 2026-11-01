"""Exercise MJPEG framing with fake HTTP, never a live camera or robot."""

from email.message import Message
from http.client import HTTPResponse
import io
import unittest
from unittest.mock import patch
from urllib.error import URLError

import cv2
import numpy as np

from vision.mjpeg import MAX_FRAME_BYTES, MJPEGStream


class FakeResponse:
    def __init__(self, body, max_read=None, content_type=None, read_error=None):
        self.body = io.BytesIO(body)
        self.headers = Message()
        self.headers["Content-Type"] = content_type or 'multipart/x-mixed-replace; boundary="camera"'
        self.max_read = max_read
        self.read_error = read_error
        self.read_calls = 0
        self.close_calls = 0

    def readline(self, size):
        return self.body.readline(size)

    def read(self, size):
        self.read_calls += 1
        if self.read_error is not None:
            raise self.read_error
        return self.body.read(min(size, self.max_read) if self.max_read else size)

    def close(self):
        self.close_calls += 1


def multipart(images):
    body = b""
    for image in images:
        body += (b"\r\n--camera\r\nContent-Type: image/jpeg\r\nContent-Length: "
                 + str(len(image)).encode("ascii") + b"\r\n\r\n" + image)
    return body + b"\r\n--camera--\r\n"


class MJPEGTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((120, 160, 3), (30, 110, 210), np.uint8)
        ok, encoded = cv2.imencode(".jpg", self.image)
        self.assertTrue(ok)
        self.jpeg = encoded.tobytes()

    def open_stream(self, response):
        opening = patch("vision.mjpeg.urllib.request.urlopen", return_value=response)
        factory = opening.start()
        self.addCleanup(opening.stop)
        stream = MJPEGStream("http://camera.invalid:81/stream")
        self.addCleanup(stream.release)
        return stream, factory

    def test_complete_jpeg_with_short_body_reads(self):
        response = FakeResponse(multipart([self.jpeg]), max_read=7)
        stream, factory = self.open_stream(response)
        self.assertTrue(stream.isOpened())
        ok, frame = stream.read()
        self.assertTrue(ok)
        self.assertEqual(frame.shape, self.image.shape)
        self.assertLess(np.abs(frame.astype(int) - self.image).mean(), 5)
        self.assertGreater(response.read_calls, 1)
        factory.assert_called_once()

    def test_http_chunks_split_across_jpeg_headers_and_bad_frame_do_not_mix_images(self):
        damaged = bytearray(self.jpeg)
        marker = damaged.find(b"\xff\xdb")
        damaged[marker + 2:marker + 4] = (65).to_bytes(2, "big")
        payload = multipart([bytes(damaged), self.jpeg])
        # HTTP chunks deliberately cut through MIME lines and JPEG marker bytes.
        chunks = [payload[index:index + 7] for index in range(0, len(payload), 7)]
        body = b"".join(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n" for chunk in chunks)
        wire = (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n"
                b"Content-Type: multipart/x-mixed-replace; boundary=camera\r\n\r\n"
                + body + b"0\r\n\r\n")

        class FakeSocket:
            def makefile(self, *args):
                return io.BufferedReader(io.BytesIO(wire))

        response = HTTPResponse(FakeSocket())
        response.begin()
        stream, factory = self.open_stream(response)
        ok, frame = stream.read()
        self.assertTrue(ok)
        self.assertEqual(frame.shape, self.image.shape)
        self.assertEqual(stream.skipped_frames, 1)
        self.assertTrue(stream.isOpened())
        factory.assert_called_once()

    def test_malformed_dqt_is_skipped_before_decoder_and_next_frame_survives(self):
        damaged = bytearray(self.jpeg)
        marker = damaged.find(b"\xff\xdb")
        self.assertGreater(marker, 0)
        damaged[marker + 2:marker + 4] = (65).to_bytes(2, "big")
        response = FakeResponse(multipart([bytes(damaged), self.jpeg]))
        stream, factory = self.open_stream(response)
        real_decode = cv2.imdecode
        with patch("vision.mjpeg.cv2.imdecode", wraps=real_decode) as decode:
            ok, frame = stream.read()
        self.assertTrue(ok)
        self.assertIsNotNone(frame)
        self.assertEqual(stream.skipped_frames, 1)
        self.assertEqual(decode.call_count, 1)
        self.assertTrue(stream.isOpened())
        factory.assert_called_once()

    def test_invalid_dqt_precision_is_skipped(self):
        damaged = bytearray(self.jpeg)
        marker = damaged.find(b"\xff\xdb")
        damaged[marker + 4] = 0x20
        stream, _ = self.open_stream(FakeResponse(multipart([bytes(damaged), self.jpeg])))
        ok, _ = stream.read()
        self.assertTrue(ok)
        self.assertEqual(stream.skipped_frames, 1)

    def test_decode_none_skips_image_without_reconnecting(self):
        response = FakeResponse(multipart([self.jpeg, self.jpeg]))
        stream, factory = self.open_stream(response)
        with patch("vision.mjpeg.cv2.imdecode", side_effect=[None, self.image.copy()]):
            ok, frame = stream.read()
        self.assertTrue(ok)
        self.assertIsNotNone(frame)
        self.assertEqual(stream.skipped_frames, 1)
        factory.assert_called_once()

    def test_truncated_image_closes_transport_without_partial_frame(self):
        body = (b"--camera\r\nContent-Length: " + str(len(self.jpeg)).encode()
                + b"\r\n\r\n" + self.jpeg[:-20])
        response = FakeResponse(body, max_read=23)
        stream, _ = self.open_stream(response)
        self.assertEqual(stream.read(), (False, None))
        self.assertFalse(stream.isOpened())
        self.assertIn("during a JPEG", stream.last_error)
        self.assertEqual(response.close_calls, 1)

    def test_invalid_and_oversized_lengths_fail_before_body_read(self):
        for length in (b"-1", b"0", b"abc", str(MAX_FRAME_BYTES + 1).encode()):
            with self.subTest(length=length):
                response = FakeResponse(b"--camera\r\nContent-Length: " + length + b"\r\n\r\n")
                with patch("vision.mjpeg.urllib.request.urlopen", return_value=response):
                    stream = MJPEGStream("http://camera.invalid/stream")
                self.assertEqual(stream.read(), (False, None))
                self.assertEqual(response.read_calls, 0)
                self.assertEqual(response.close_calls, 1)

    def test_five_consecutive_invalid_frames_close_connection(self):
        response = FakeResponse(multipart([b"\xff\xd8\xff\xd9"] * 5 + [self.jpeg]))
        stream, _ = self.open_stream(response)
        self.assertEqual(stream.read(), (False, None))
        self.assertEqual(stream.skipped_frames, 5)
        self.assertFalse(stream.isOpened())
        self.assertEqual(response.close_calls, 1)

    def test_timeout_during_image_read_closes_connection(self):
        response = FakeResponse(multipart([self.jpeg]), read_error=TimeoutError())
        stream, _ = self.open_stream(response)
        self.assertEqual(stream.read(), (False, None))
        self.assertIn("timed out", stream.last_error)
        self.assertEqual(response.close_calls, 1)

    def test_connection_error_leaves_closed_stream(self):
        with patch("vision.mjpeg.urllib.request.urlopen", side_effect=URLError(TimeoutError())), patch("builtins.print"):
            stream = MJPEGStream("http://camera.invalid/stream")
        self.assertFalse(stream.isOpened())
        self.assertEqual(stream.read(), (False, None))
        self.assertIn("timed out", stream.last_error)

    def test_release_is_idempotent(self):
        response = FakeResponse(multipart([self.jpeg]))
        stream, _ = self.open_stream(response)
        stream.release()
        stream.release()
        self.assertFalse(stream.isOpened())
        self.assertEqual(stream.read(), (False, None))
        self.assertEqual(response.close_calls, 1)

    def test_bad_content_type_boundary_and_long_header_are_bounded(self):
        for content_type in ("text/plain", "multipart/x-mixed-replace; boundary=" + "x" * 71):
            with self.subTest(content_type=content_type), patch("builtins.print"):
                response = FakeResponse(b"", content_type=content_type)
                with patch("vision.mjpeg.urllib.request.urlopen", return_value=response):
                    stream = MJPEGStream("http://camera.invalid/stream")
                self.assertFalse(stream.isOpened())
                self.assertEqual(response.close_calls, 1)
        response = FakeResponse(b"--camera\r\nX-Long: " + b"x" * 2000 + b"\r\n\r\n")
        stream, _ = self.open_stream(response)
        self.assertEqual(stream.read(), (False, None))
        self.assertEqual(response.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
