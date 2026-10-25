##Detect colored blobs and confirm consecutive same-color results.

import cv2
import numpy as np

from vision.lighting import compensate_lighting
from vision.white_sheet import find_sheet_region

# Area ratios reject image noise without assuming a physical flag size.
# Shape filtering stays optional when flag proportions are unknown.


def build_mask(hsv_frame, color_ranges, settings):
#Builds and cleans the color mask using morphological operations
    kernel_size=settings.get("kernel_size", 5) #size of cleanup filter
#pixels inside the first HSV range become 255, everything else 0
    lower1=np.array(color_ranges["lower1"])
    upper1=np.array(color_ranges["upper1"])
    mask=cv2.inRange(hsv_frame, lower1, upper1)
#colors like red wrap around HSV so they need a second range
    if "lower2" in color_ranges and "upper2" in color_ranges:
        lower2=np.array(color_ranges["lower2"])
        upper2=np.array(color_ranges["upper2"])
        mask2=cv2.inRange(hsv_frame, lower2, upper2)
#combines both ranges: a pixel is counted when it's in either one
        mask=cv2.bitwise_or(mask, mask2)

    kernel=np.ones((kernel_size, kernel_size), np.uint8)
#OPEN removes salt (noise) and CLOSE fills small black holes inside white areas
    mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask #mask is cleaned and sent


def find_largest_blob(mask, settings, sheet_region=None):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Use an area ratio so the noise threshold scales with image resolution.
    image_height, image_width = mask.shape[:2]
    image_area = image_height * image_width
    min_area = settings["min_flag_area_ratio"] * image_area

    # Keep shape filtering optional because the flag proportions are unknown.
    aspect_limits = settings.get("aspect_ratio_range")

    valid_blobs = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if aspect_limits is not None:
            min_ar, max_ar = aspect_limits
            aspect_ratio = w / h

            if not (min_ar <= aspect_ratio <= max_ar):
                continue

        if sheet_region is not None:
            # Check the complete blob so clipping an outside object cannot
            # produce a small false flag or move its reported center.
            blob_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(
                blob_mask, [contour], -1, 255, cv2.FILLED, offset=(-x, -y)
            )
            inside = cv2.bitwise_and(
                blob_mask, sheet_region[y:y + h, x:x + w]
            )
            if cv2.countNonZero(inside) < 0.9 * cv2.countNonZero(blob_mask):
                continue

        valid_blobs.append({
            "area": area,
            "box": (x, y, w, h)
        })

    if not valid_blobs:
        return None

    return max(valid_blobs, key=lambda blob: blob["area"])


def detect_flag(frame, settings):
    """Detects flag color, bounding box, area, and center."""
#blur first to smooth out camera noise
    blur_ksize=settings.get("blur_kernel_size", 5)
    blurred_frame=cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), 0)

#convert from BGR to HSV, better for color detection
    hsv=cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    # Locate the sheet before brightness correction, using neutral pixels.
    sheet_region = find_sheet_region(hsv, settings)
    if sheet_region is not None and cv2.countNonZero(sheet_region) == 0:
        return {"detected": False, "sheet_found": False}
    # Optionally adjust brightness before checking the configured color ranges.
    hsv = compensate_lighting(hsv, settings)
#variables that will hold the winning color and its blob
    best_color=None
    best_blob=None
    # Search all configured profiles; no operator-selected color is required.
    for color_name, ranges in settings["colors"].items():
        mask = build_mask(hsv, ranges, settings)
        blob=find_largest_blob(mask, settings, sheet_region)
#mask is made first  to find the best flag shaped blob in that mask

        if blob is not None:
            if best_blob is None or blob["area"]>best_blob["area"]:
                best_blob=blob
                best_color=color_name

    if best_color is None:
        return {"detected": False}

    x, y, w, h= best_blob["box"] #center
    center_x=x+w//2
    center_y=y+h//2

    return {
        "detected": True,
        "color": best_color,
        "area": best_blob["area"],
        "box": best_blob["box"],
        "center": (center_x, center_y)
    }


class FlagStabilizer:
#Handles consecutive frame confirmation to ensure reliable detection
    def __init__(self, required_consecutive_frames=4): #4 frames in a row must show the same color before we trust it
        self.required_frames=required_consecutive_frames
        self.current_color=None
        self.consecutive_count=0

    def update(self, detection_result):
        if detection_result["detected"]:
            detected_color=detection_result["color"]

            if detected_color==self.current_color:
                self.consecutive_count+=1
            else:
                self.current_color=detected_color #different color
                self.consecutive_count = 1
        else:
            self.current_color=None
            self.consecutive_count=0
            return {"confirmed": False}

        if self.consecutive_count >= self.required_frames: #confirm and pass the detection data
            return {
                "confirmed": True,
                "color": self.current_color,
                "center": detection_result["center"],
                "box": detection_result["box"],
                "area": detection_result["area"]
            }
        return {"confirmed": False}
