"""Check brightness correction without requiring a live camera."""

import copy
import unittest

import cv2
import numpy as np

from vision.detector import detect_flag
from vision.lighting import compensate_lighting
from vision.settings import load_settings


class LightingTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings()
        self.settings["lighting_compensation"] = {
            "enabled": True, "clip_limit": 2.0, "tile_grid_size": 8
        }

    def test_disabled_or_missing_option_preserves_image(self):
        hsv = np.full((240, 320, 3), (60, 150, 45), dtype=np.uint8)
        for settings in ({}, {"lighting_compensation": {"enabled": False}}):
            with self.subTest(settings=settings):
                np.testing.assert_array_equal(
                    compensate_lighting(hsv, settings), hsv
                )

    def test_only_brightness_changes_without_modifying_input(self):
        hsv = np.full((240, 320, 3), (60, 150, 45), dtype=np.uint8)
        original = hsv.copy()
        corrected = compensate_lighting(hsv, self.settings)
        np.testing.assert_array_equal(hsv, original)
        np.testing.assert_array_equal(corrected[:, :, :2], hsv[:, :, :2])
        self.assertTrue(np.any(corrected[:, :, 2] != hsv[:, :, 2]))

    def test_dim_green_can_cross_brightness_threshold(self):
        # Use a threshold of 50 to reproduce the earlier dark-green failure.
        self.settings["colors"]["green"]["lower1"] = [35, 40, 50]
        hsv = np.full((240, 320, 3), (0, 0, 150), dtype=np.uint8)
        hsv[70:150, 110:210] = (60, 180, 45)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        disabled = copy.deepcopy(self.settings)
        disabled["lighting_compensation"]["enabled"] = False
        self.assertFalse(detect_flag(frame, disabled)["detected"])
        result = detect_flag(frame, self.settings)
        self.assertTrue(result["detected"])
        self.assertEqual(result["color"], "green")

    def test_neutral_background_and_shadow_do_not_become_a_color(self):
        for brightness in (0, 40, 150, 255):
            with self.subTest(brightness=brightness):
                frame = np.full((240, 320, 3), brightness, dtype=np.uint8)
                frame[70:150, 110:210] = brightness // 2
                self.assertFalse(detect_flag(frame, self.settings)["detected"])


if __name__ == "__main__":
    unittest.main()
