"""Reject room objects while preserving full flag coordinates on the sheet."""

import unittest

import cv2
import numpy as np

from vision.detector import detect_flag
from vision.settings import load_settings


class SheetRegionTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings()

    def make_scene(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[35:220, 150:305] = 210
        return frame

    def test_larger_green_outside_does_not_hide_red_on_sheet(self):
        frame = self.make_scene()
        frame[60:175, 10:125] = (0, 255, 0)
        frame[100:155, 200:255] = (0, 0, 255)
        result = detect_flag(frame, self.settings)
        self.assertTrue(result["detected"])
        self.assertEqual(result["color"], "red")
        self.assertAlmostEqual(result["center"][0], 227, delta=2)
        self.assertAlmostEqual(result["center"][1], 127, delta=2)

    def test_green_outside_blank_sheet_is_rejected(self):
        frame = self.make_scene()
        frame[65:175, 10:120] = (0, 255, 0)
        self.assertFalse(detect_flag(frame, self.settings)["detected"])

    def test_smaller_green_inside_wins_over_larger_green_outside(self):
        frame = self.make_scene()
        frame[60:175, 10:125] = (0, 255, 0)
        frame[100:155, 200:255] = (0, 120, 0)
        result = detect_flag(frame, self.settings)
        self.assertTrue(result["detected"])
        self.assertEqual(result["color"], "green")
        self.assertGreater(result["center"][0], 190)

    def test_no_sheet_clearly_reports_missing_sheet(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[70:150, 110:210] = (0, 150, 0)
        result = detect_flag(frame, self.settings)
        self.assertFalse(result["detected"])
        self.assertFalse(result["sheet_found"])

    def test_object_straddling_sheet_edge_is_not_clipped_into_flag(self):
        frame = self.make_scene()
        frame[80:170, 110:185] = (0, 180, 0)
        self.assertFalse(detect_flag(frame, self.settings)["detected"])

    def test_rotated_sheet_keeps_flag_and_image_coordinates(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        corners = cv2.boxPoints(((190, 125), (180, 150), 15)).astype(np.int32)
        cv2.fillConvexPoly(frame, corners, (210, 210, 210))
        frame[105:145, 170:210] = (0, 130, 0)
        result = detect_flag(frame, self.settings)
        self.assertTrue(result["detected"])
        self.assertEqual(result["color"], "green")
        self.assertAlmostEqual(result["center"][0], 190, delta=2)
        self.assertAlmostEqual(result["center"][1], 125, delta=2)

    def test_white_speckles_do_not_unlock_room_search(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[70:150, 110:210] = (0, 150, 0)
        frame[20:23, 20:23] = 255
        frame[210:214, 280:284] = 255
        self.assertFalse(detect_flag(frame, self.settings)["detected"])

    def test_filter_can_be_disabled_explicitly(self):
        self.settings["white_sheet"]["enabled"] = False
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[70:150, 110:210] = (0, 150, 0)
        result = detect_flag(frame, self.settings)
        self.assertTrue(result["detected"])
        self.assertEqual(result["color"], "green")


if __name__ == "__main__":
    unittest.main()
