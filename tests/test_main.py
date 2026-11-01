"""Check the main loop and cleanup without a camera or robot."""

import contextlib
import io
import unittest
from unittest.mock import MagicMock, patch

import test as app


class MainLoopTests(unittest.TestCase):
    def setUp(self):
        # Offline main-loop tests must never send motor-control UDP packets.
        socket_patch = patch.object(app.socket, "socket")
        self.socket_factory = socket_patch.start()
        self.addCleanup(socket_patch.stop)
        sleep_patch = patch.object(app.time, "sleep")
        sleep_patch.start()
        self.addCleanup(sleep_patch.stop)
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def sent_messages(self):
        return [call.args[0] for call in self.socket_factory.return_value.sendto.call_args_list]

    def test_r_requires_four_fresh_detections_without_reopening_camera(self):
        for reset_key in ("r", "R"):
            with self.subTest(reset_key=reset_key):
                image = app.np.full((240, 320, 3), 255, dtype=app.np.uint8)
                image[70:150, 110:210] = (0, 0, 255)
                capture = MagicMock()
                capture.isOpened.return_value = True
                capture.read.side_effect = [(True, image.copy()) for _ in range(8)]
                with patch.object(app, "open_camera", return_value=capture) as opening, patch.object(
                    app.cv2, "waitKey",
                    side_effect=[-1] * 3 + [ord(reset_key)] + [-1] * 3 + [ord("q")]
                ), patch.object(app.cv2, "imshow"), patch.object(
                    app.cv2, "destroyAllWindows"
                ):
                    app.main()
                # Confirm on frame 4, press reset, then confirm again on frame 8.
                messages = self.sent_messages()[-8:]
                self.assertEqual([m.startswith(b"DET,") for m in messages],
                                 [False, False, False, True, False, False, False, True])
                opening.assert_called_once()
                capture.release.assert_called_once()

    def test_main_closes_failed_stream_before_one_reconnect_attempt(self):
        # Bound the reconnect scenario so the test cannot loop indefinitely.
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        failed_reopen = MagicMock()
        failed_reopen.isOpened.return_value = False
        with patch.object(app, "MAX_RECONNECT_ATTEMPTS", 1), patch.object(
            app, "open_camera", side_effect=[capture, failed_reopen]
        ) as opening, patch.object(app.cv2, "waitKey", return_value=-1), patch.object(
            app.cv2, "imshow"
        ), patch.object(app.cv2, "destroyAllWindows"):
            app.main()
        self.assertEqual(self.sent_messages(), [b"NONE\n"] * app.MAX_CONSECUTIVE_FAILURES)
        self.assertEqual(opening.call_count, 2)
        capture.release.assert_called_once()
        failed_reopen.release.assert_called_once()
        self.socket_factory.return_value.close.assert_called_once()

    def test_main_retries_temporary_reopen_failure_then_resumes(self):
        lost = MagicMock()
        lost.isOpened.return_value = True
        lost.read.return_value = (False, None)
        failed_reopen = MagicMock()
        failed_reopen.isOpened.return_value = False
        recovered = MagicMock()
        recovered.isOpened.return_value = True
        frame = app.np.full((240, 320, 3), 255, dtype=app.np.uint8)
        recovered.read.return_value = (True, frame)
        with patch.object(app, "open_camera", side_effect=[lost, failed_reopen, recovered]) as opening, patch.object(
            app.cv2, "waitKey", side_effect=[-1] * app.MAX_CONSECUTIVE_FAILURES + [ord("q")]
        ), patch.object(app.cv2, "imshow"), patch.object(app.cv2, "destroyAllWindows"):
            app.main()
        self.assertEqual(opening.call_count, 3)
        lost.release.assert_called_once()
        failed_reopen.release.assert_called_once()
        recovered.release.assert_called_once()
        recovered.read.assert_called_once()

    def test_main_confirms_after_four_real_detections(self):
        # Exercise the real detector and stabilizer without a physical camera.
        image = app.np.full((240, 320, 3), 255, dtype=app.np.uint8)
        image[40:120, 50:150] = (0, 0, 255)
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.side_effect = [(True, image.copy()) for _ in range(7)]
        with patch.object(app, "open_camera", return_value=capture), patch.object(
            app.cv2, "waitKey", side_effect=[-1] * 6 + [ord("q")]
        ), patch.object(app.cv2, "imshow"), patch.object(
            app.cv2, "destroyAllWindows"
        ):
            app.main()
        confirmed = [m.startswith(b"DET,RED,") for m in self.sent_messages()]
        self.assertEqual(confirmed, [False] * 3 + [True] * 4)
        capture.release.assert_called_once()

    def test_preview_searches_then_confirms(self):
        for width, height in ((160, 120), (320, 240)):
            with self.subTest(resolution=(width, height)):
                blank = app.np.full((height, width, 3), 210, dtype=app.np.uint8)
                target = blank.copy()
                target[height // 3:2 * height // 3, width // 3:2 * width // 3] = (0, 0, 255)
                capture = MagicMock()
                capture.isOpened.return_value = True
                capture.read.side_effect = [(True, image.copy()) for image in [blank] + [target] * 4]
                with patch.object(app, "open_camera", return_value=capture), patch.object(
                    app.cv2, "waitKey", side_effect=[-1] * 4 + [ord("q")]
                ), patch.object(app.cv2, "imshow"), patch.object(
                    app.cv2, "destroyAllWindows"
                ), patch.object(app.cv2, "putText", wraps=app.cv2.putText) as drawing:
                    app.main()
                status_calls = [call for call in drawing.call_args_list
                                if call.args[1].startswith(("Searching for", "Confirming"))]
                self.assertEqual([call.args[1] for call in status_calls], [
                    "Searching for flag...",
                    "Confirming red flag...", "Confirming red flag...", "Confirming red flag...",
                ])
                for call in status_calls:
                    size = app.cv2.getTextSize(call.args[1], call.args[3], call.args[4], call.args[6])[0][0]
                    self.assertLessEqual(size + call.args[2][0], width)

    def test_corrupt_image_gap_requires_fresh_confirmation_before_robot_detection(self):
        frame = app.np.full((240, 320, 3), 255, dtype=app.np.uint8)
        frame[70:150, 110:210] = (0, 0, 255)
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.skipped_frames = 0

        def read_frame():
            capture.skipped_frames = 0 if capture.read.call_count <= 4 else 1
            return True, frame.copy()

        capture.read.side_effect = read_frame
        with patch.object(app, "open_camera", return_value=capture), patch.object(
            app.cv2, "waitKey", side_effect=[-1] * 7 + [ord("q")]
        ), patch.object(app.cv2, "imshow"), patch.object(app.cv2, "destroyAllWindows"):
            app.main()
        messages = self.sent_messages()
        confirmed = [message.startswith(b"DET,") for message in messages]
        self.assertEqual(confirmed, [False, False, False, True, False, False, False, True])

    def test_main_does_not_reopen_while_previous_reader_is_still_alive(self):
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        capture.release.return_value = False
        with patch.object(app, "open_camera", return_value=capture) as opening, patch.object(
            app.cv2, "waitKey", return_value=-1
        ), patch.object(app.cv2, "imshow"), patch.object(app.cv2, "destroyAllWindows"):
            app.main()
        opening.assert_called_once()
        self.assertIn("Previous camera reader is still closing", self.output.getvalue())

    def test_main_closes_socket_if_camera_open_fails(self):
        with patch.object(
            app, "open_camera", side_effect=RuntimeError("camera backend failed")
        ), patch.object(app.cv2, "destroyAllWindows"):
            with self.assertRaises(RuntimeError):
                app.main()
        self.socket_factory.return_value.close.assert_called_once()

    def test_c_does_not_open_selector_or_interrupt_automatic_detection(self):
        # Both colors must confirm automatically, even if the removed key is pressed.
        red = app.np.full((240, 320, 3), 255, dtype=app.np.uint8)
        red[40:120, 50:150] = (0, 0, 255)
        blue = app.np.full((240, 320, 3), 255, dtype=app.np.uint8)
        blue[40:120, 50:150] = (255, 0, 0)
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.side_effect = [
            (True, image.copy()) for image in [red] * 4 + [blue] * 4
        ]
        with patch.object(app, "open_camera", return_value=capture), patch.object(
            app.cv2, "waitKey", side_effect=[ord("c")] + [-1] * 6 + [ord("q")]
        ), patch.object(app.cv2, "selectROI") as selector, patch.object(
            app.cv2, "imshow"
        ), patch.object(app.cv2, "destroyAllWindows"):
            app.main()
        selector.assert_not_called()
        messages = self.sent_messages()
        self.assertTrue(messages[3].startswith(b"DET,RED,"))
        self.assertTrue(messages[7].startswith(b"DET,BLUE,"))
        capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
