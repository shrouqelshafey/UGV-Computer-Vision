"""Check announcements and cleanup without a camera, LCD or Serial device."""

import contextlib
import io
import sys
import unittest
from unittest.mock import MagicMock, patch

import test as app
from communication.lcd_announcer import FlagAnnouncer, announce_on_lcd


class AnnouncementTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def test_terminal_mode_needs_no_serial_package(self):
        # Disabled hardware must work even if pySerial cannot be imported.
        with patch.dict(sys.modules, {"serial": None}):
            announcer = FlagAnnouncer()
            for _ in range(20):
                announcer.update({"confirmed": True, "color": "red"})
        self.assertEqual(self.output.getvalue().count("COLOR: RED"), 1)

    def test_ten_unconfirmed_updates_allow_another_announcement(self):
        announcer = FlagAnnouncer()
        red = {"confirmed": True, "color": "red"}
        announcer.update(red)
        for _ in range(9):
            announcer.update({"confirmed": False})
        announcer.update(red)
        self.assertEqual(self.output.getvalue().count("COLOR: RED"), 1)
        for _ in range(10):
            announcer.update({"confirmed": False})
        announcer.update(red)
        self.assertEqual(self.output.getvalue().count("COLOR: RED"), 2)

    def test_color_change_allows_a_new_announcement(self):
        announcer = FlagAnnouncer()
        announcer.update({"confirmed": True, "color": "red"})
        announcer.update({"confirmed": True, "color": "blue"})
        announcer.update({"confirmed": True, "color": "blue"})
        self.assertEqual(self.output.getvalue().count("FLAG DETECTED"), 2)

    def test_serial_message_matches_teammates_receiver_format(self):
        port = MagicMock()
        port.write.return_value = len(b"FLAG_DETECTED,RED\n")
        announce_on_lcd("red", port)
        port.write.assert_called_once_with(b"FLAG_DETECTED,RED\n")

    def test_write_failure_closes_serial_and_keeps_terminal_mode(self):
        announcer = FlagAnnouncer()
        port = MagicMock()
        port.write.side_effect = OSError("device disconnected")
        announcer.esp32 = port
        announcer.update({"confirmed": True, "color": "red"})
        port.close.assert_called_once()
        self.assertIsNone(announcer.esp32)
        announcer.update({"confirmed": True, "color": "blue"})
        self.assertIn("COLOR: BLUE", self.output.getvalue())

    def test_optional_serial_setup_and_close(self):
        serial_module = MagicMock()
        with patch.dict(sys.modules, {"serial": serial_module}), patch("time.sleep"):
            announcer = FlagAnnouncer(use_esp32=True, port="TEST_PORT")
        serial_module.Serial.assert_called_once_with(
            "TEST_PORT", 115200, timeout=1, write_timeout=1
        )
        announcer.close()
        serial_module.Serial.return_value.close.assert_called_once()

    def test_reset_allows_same_color_without_closing_serial(self):
        announcer = FlagAnnouncer()
        port = MagicMock()
        message = b"FLAG_DETECTED,RED\n"
        port.write.return_value = len(message)
        announcer.esp32 = port
        red = {"confirmed": True, "color": "red"}
        announcer.update(red)
        announcer.reset()
        announcer.update(red)
        self.assertEqual(port.write.call_count, 2)
        port.close.assert_not_called()

    def test_r_requires_four_fresh_detections_without_reopening_camera(self):
        for reset_key in ("r", "R"):
            with self.subTest(reset_key=reset_key):
                image = app.np.full((240, 320, 3), 255, dtype=app.np.uint8)
                image[70:150, 110:210] = (0, 0, 255)
                capture = MagicMock()
                capture.isOpened.return_value = True
                capture.read.side_effect = [(True, image.copy()) for _ in range(8)]
                announced_at_frames = []
                with patch.object(app, "open_camera", return_value=capture) as opening, patch.object(
                    app.cv2, "waitKey",
                    side_effect=[-1] * 3 + [ord(reset_key)] + [-1] * 3 + [ord("q")]
                ), patch.object(app.cv2, "imshow"), patch.object(
                    app.cv2, "destroyAllWindows"
                ), patch(
                    "communication.lcd_announcer.announce_on_lcd",
                    side_effect=lambda *args: announced_at_frames.append(capture.read.call_count)
                ):
                    app.main()
                # Confirm on frame 4, press reset, then confirm again on frame 8.
                self.assertEqual(announced_at_frames, [4, 8])
                opening.assert_called_once()
                capture.release.assert_called_once()

    def test_main_counts_failed_reads_without_reopening(self):
        # A camera outage must not bring back the removed reconnection feature.
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        with patch.object(app, "FlagAnnouncer") as factory, patch.object(
            app, "open_camera", return_value=capture
        ) as opening, patch.object(app.cv2, "waitKey", return_value=-1), patch.object(
            app.cv2, "imshow"
        ), patch.object(app.cv2, "destroyAllWindows"):
            app.main()
        self.assertEqual(factory.return_value.update.call_count, 30)
        opening.assert_called_once()
        capture.release.assert_called_once()
        factory.return_value.close.assert_called_once()

    def test_main_announces_once_after_four_real_detections(self):
        # Exercise the real detector and stabilizer without a physical camera.
        # Match the competition setup: a non-white flag on a white sheet.
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
        self.assertEqual(self.output.getvalue().count("COLOR: RED"), 1)
        capture.release.assert_called_once()

    def test_main_closes_optional_serial_if_camera_open_fails(self):
        with patch.object(app, "FlagAnnouncer") as factory, patch.object(
            app, "open_camera", side_effect=RuntimeError("camera backend failed")
        ), patch.object(app.cv2, "destroyAllWindows"):
            with self.assertRaises(RuntimeError):
                app.main()
        factory.return_value.close.assert_called_once()

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
        self.assertEqual(self.output.getvalue().count("COLOR: RED"), 1)
        self.assertEqual(self.output.getvalue().count("COLOR: BLUE"), 1)
        capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
