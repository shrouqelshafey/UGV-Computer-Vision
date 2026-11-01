"""Optional local brightness correction before HSV color detection."""

from functools import lru_cache

import cv2


@lru_cache(maxsize=1)
def _get_clahe(clip_limit, grid_size):
    # Reuse the processor instead of constructing it for every camera frame.
    return cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(grid_size, grid_size)
    )


def compensate_lighting(hsv_frame, settings):
    """Adjust V only; preserve the hue and saturation used by color masks."""
    options = settings.get("lighting_compensation", {})
    if not options.get("enabled", False):
        return hsv_frame

    processor = _get_clahe(
        float(options.get("clip_limit", 2.0)),
        options.get("tile_grid_size", 8),
    )
    # Copy so the original HSV image remains available for comparisons.
    corrected = hsv_frame.copy()
    corrected[:, :, 2] = processor.apply(hsv_frame[:, :, 2])
    return corrected
