"""
Image preprocessing placeholders for future work.

MVP note: For now, we assume images are already cropped before upload.
These functions return the original image unchanged but keep the call
sites stable so we can drop in real implementations later without
touching the rest of the code.
"""
from typing import Any
import importlib

# Optional dependency import for OpenCV
cv2: Any
try:
    cv2 = importlib.import_module("cv2")
except Exception:
    cv2 = None  # type: ignore[assignment]


def crop_receipt(image_path: str):
    """Return the image as-is (placeholder for future receipt crop/warp)."""
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for image cropping but is not installed.")
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Failed to read image for crop placeholder")
    return img


def make_receipt_preview(image_path: str):
    """Return the image as-is for preview (placeholder)."""
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for image preview but is not installed.")
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Failed to read image for preview placeholder")
    return img
