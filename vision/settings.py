"""Load the shared JSON settings when the application starts."""

import json
import math
from pathlib import Path


def load_settings(path=None):
    """Load and validate settings from JSON."""
    # Resolve the default file from the project, not the terminal directory.
    if path is None:
        path = Path(__file__).resolve().parent.parent / "settings.json"
    with open(path, "r", encoding="utf-8") as file:
        settings=json.load(file)
#Red wraps around in HSV, so lower/upper2 must come as a pair
    for color_name, ranges in settings["colors"].items():
        if ("lower2" in ranges) != ("upper2" in ranges):
            raise ValueError(f"Color '{color_name}' needs both lower2 and upper2")
#Gaussian blur crashes if the size is even
#Works by sliding an odd sized square window  - center pixel and same number of neighbors on each side
    if "blur_kernel_size" in settings:
        blur_ksize= settings["blur_kernel_size"]
        if not isinstance(blur_ksize, int) or blur_ksize <= 0 or blur_ksize % 2 == 0:
            raise ValueError("'blur_kernel_size' must be a positive odd integer.")
    # Reject invalid lighting options before OpenCV starts processing frames.
    lighting = settings.get("lighting_compensation", {})
    if not isinstance(lighting, dict):
        raise ValueError("'lighting_compensation' must be an object.")
    if not isinstance(lighting.get("enabled", False), bool):
        raise ValueError("Lighting 'enabled' must be true or false.")
    clip_limit = lighting.get("clip_limit", 2.0)
    if (type(clip_limit) not in (int, float)
            or not math.isfinite(clip_limit) or clip_limit <= 0):
        raise ValueError("Lighting 'clip_limit' must be a positive finite number.")
    grid_size = lighting.get("tile_grid_size", 8)
    if type(grid_size) is not int or not 1 <= grid_size <= 32:
        raise ValueError("Lighting 'tile_grid_size' must be an integer from 1 to 32.")
    # Sheet filtering needs valid pixel thresholds and a relative minimum size.
    sheet = settings.get("white_sheet", {})
    if not isinstance(sheet, dict):
        raise ValueError("'white_sheet' must be an object.")
    if not isinstance(sheet.get("enabled", False), bool):
        raise ValueError("White-sheet 'enabled' must be true or false.")
    for key, default in (("max_saturation", 40), ("min_brightness", 60)):
        value = sheet.get(key, default)
        if type(value) is not int or not 0 <= value <= 255:
            raise ValueError(f"White-sheet '{key}' must be an integer from 0 to 255.")
    area_ratio = sheet.get("min_area_ratio", 0.02)
    if type(area_ratio) not in (int, float) or not 0 < area_ratio <= 1:
        raise ValueError("White-sheet 'min_area_ratio' must be greater than 0 and at most 1.")
    return settings
