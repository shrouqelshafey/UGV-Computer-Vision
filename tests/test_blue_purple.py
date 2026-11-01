"""Check dim and muted cool colors."""

import unittest

import cv2
import numpy as np

from vision.detector import detect_flag
from vision.settings import load_settings


class BluePurpleTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings()

    def make_sample(self, color, dark_border=False):
        # Place the sample on a neutral gray background.
        hsv = np.full((240, 320, 3), (0, 0, 180), dtype=np.uint8)
        if dark_border:
            hsv[65:155, 105:215] = (0, 0, 0)
        hsv[70:150, 110:210] = color
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def test_muted_blue_and_dim_blue_purple_are_detected(self):
        samples = [
            ("blue", (110, 80, 90)),
            ("blue", (120, 160, 35)),
            ("purple", (140, 100, 35)),
            ("purple", (150, 70, 90)),
        ]
        for expected, hsv in samples:
            with self.subTest(expected=expected, hsv=hsv):
                result = detect_flag(self.make_sample(hsv), self.settings)
                self.assertTrue(result["detected"])
                self.assertEqual(result["color"], expected)

    def test_purple_with_dark_border_is_not_claimed_by_blue(self):
        # At hue 135, the old overlapping profiles could both match equally.
        frame = self.make_sample((135, 255, 180), dark_border=True)
        result = detect_flag(frame, self.settings)
        self.assertTrue(result["detected"])
        self.assertEqual(result["color"], "purple")


if __name__ == "__main__":
    unittest.main()
