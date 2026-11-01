"""Check fresh-frame delivery and reader ownership without a physical camera."""

import queue
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from vision.camera import LatestFrameCamera


class CameraTests(unittest.TestCase):
    def make_camera(self):
        incoming = queue.Queue()
        backend = MagicMock()
        backend.isOpened.return_value = True
        backend.read.side_effect = incoming.get
        with patch("vision.camera.MJPEGStream", return_value=backend):
            camera = LatestFrameCamera("test-stream")

        def cleanup():
            # Unblock any fake read so reader cleanup completes without a device.
            camera._stop.set()
            incoming.put((False, None))
            camera.release()
        self.addCleanup(cleanup)
        return camera, incoming, backend

    def wait_for_pending(self, camera, predicate):
        with camera._condition:
            self.assertTrue(camera._condition.wait_for(predicate, timeout=2.0))

    def test_slow_consumer_gets_newest_frame_instead_of_old_queue(self):
        camera, incoming, _ = self.make_camera()
        frames = [np.full((2, 2, 3), value, np.uint8) for value in (1, 2, 3)]
        for frame in frames:
            incoming.put((True, frame))
        self.wait_for_pending(
            camera, lambda: camera._pending is not None
            and camera._pending[1] is frames[-1]
        )
        ret, frame = camera.read()
        self.assertTrue(ret)
        self.assertIs(frame, frames[-1])
        self.assertEqual(camera.dropped_frames, 2)

    def test_same_frame_is_not_delivered_twice(self):
        camera, incoming, _ = self.make_camera()
        first = np.zeros((2, 2, 3), np.uint8)
        incoming.put((True, first))
        self.assertIs(camera.read()[1], first)
        self.assertIsNone(camera._pending)
        second = first.copy()
        incoming.put((True, second))
        self.assertIs(camera.read()[1], second)

    def test_delivered_frame_reports_corrupt_images_skipped_before_it(self):
        camera, incoming, backend = self.make_camera()
        backend.skipped_frames = 2
        incoming.put((True, np.zeros((2, 2, 3), np.uint8)))
        self.assertTrue(camera.read()[0])
        self.assertEqual(camera.skipped_frames, 2)

    def test_ended_stream_clears_old_frame_and_closes_before_failure_is_reported(self):
        camera, incoming, backend = self.make_camera()
        old = np.zeros((2, 2, 3), np.uint8)
        incoming.put((True, old))
        incoming.put((False, None))
        self.wait_for_pending(camera, lambda: camera._ended)
        self.assertEqual(camera.read(), (False, None))
        self.assertEqual(camera.read(), (False, None))
        self.assertFalse(camera.isOpened())
        backend.release.assert_called_once()
        self.assertEqual(backend.read.call_count, 2)

    def test_closed_camera_does_not_return_a_pending_frame(self):
        camera, incoming, backend = self.make_camera()
        incoming.put((True, np.zeros((2, 2, 3), np.uint8)))
        self.wait_for_pending(camera, lambda: camera._pending is not None)
        camera._stop.set()
        incoming.put((False, None))
        camera.release()
        self.assertEqual(camera.read(), (False, None))
        self.assertFalse(camera.isOpened())
        backend.release.assert_called_once()

    def test_reader_error_is_reported_instead_of_hanging(self):
        backend = MagicMock()
        backend.isOpened.return_value = True
        backend.read.side_effect = RuntimeError("backend read failed")
        with patch("vision.camera.MJPEGStream", return_value=backend):
            camera = LatestFrameCamera("test-stream")
        self.addCleanup(camera.release)
        with self.assertRaisesRegex(RuntimeError, "backend read failed"):
            camera.read()

    def test_failed_open_releases_backend_without_starting_reader(self):
        backend = MagicMock()
        backend.isOpened.return_value = False
        with patch("vision.camera.MJPEGStream", return_value=backend):
            camera = LatestFrameCamera("test-stream")
        self.assertFalse(camera.isOpened())
        self.assertEqual(camera.read(), (False, None))
        camera.release()
        backend.read.assert_not_called()
        backend.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
