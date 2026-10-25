"""Keep dim green samples while excluding the observed blue-green background."""

import unittest

import cv2
import numpy as np

from vision.detector import detect_flag
from vision.settings import load_settings


class GreenThresholdTests(unittest.TestCase):
    def make_sample(self, hsv_color):
        # A neutral background isolates the color threshold from other objects.
        hsv = np.full((240, 320, 3), (0, 0, 150), dtype=np.uint8)
        hsv[70:150, 110:210] = hsv_color
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def test_dim_muted_green_still_detects(self):
        # Similar to the real sample: green hue, low saturation and brightness.
        result = detect_flag(self.make_sample((60, 60, 40)), load_settings())
        self.assertTrue(result["detected"])
        self.assertEqual(result["color"], "green")

    def test_observed_background_hue_is_rejected(self):
        # The false patch clustered around hue 81, outside the tighter range.
        result = detect_flag(self.make_sample((81, 55, 50)), load_settings())
        self.assertFalse(result["detected"])


if __name__ == "__main__":
    unittest.main()
