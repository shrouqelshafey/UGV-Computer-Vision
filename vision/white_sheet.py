"""Locate neutral sheet regions before accepting a colored flag."""

import cv2
import numpy as np


def find_sheet_region(hsv_frame, settings):
    """Return a filled search mask, or None when sheet filtering is disabled."""
    options = settings.get("white_sheet", {})
    if not options.get("enabled", False):
        return None

    # Use the original brightness so CLAHE cannot promote dark background noise.
    white = cv2.inRange(
        hsv_frame,
        np.array([0, 0, options.get("min_brightness", 60)], dtype=np.uint8),
        np.array([179, options.get("max_saturation", 40), 255], dtype=np.uint8),
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    region = np.zeros(white.shape, dtype=np.uint8)
    min_area = options.get("min_area_ratio", 0.02) * white.size
    for contour in contours:
        # Fill the boundary, including the non-white flag inside the sheet.
        # Testing only white pixels would erase the flag itself.
        boundary = cv2.convexHull(contour)
        area = cv2.contourArea(boundary)
        if area <= 0 or area < min_area:
            continue
        # Reject thin neutral streaks that would create a large artificial region.
        if cv2.contourArea(contour) < 0.25 * area:
            continue
        cv2.drawContours(region, [boundary], -1, 255, cv2.FILLED)
    # An empty mask means no sheet: do not fall back to searching the whole room.
    return region
