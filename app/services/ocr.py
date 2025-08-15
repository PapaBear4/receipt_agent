from typing import Tuple, Dict, Any, List
from PIL import Image
import pytesseract
import importlib
import numpy as np

from app.config import settings
import logging

# Module logger
logger = logging.getLogger(__name__)
if settings.DEBUG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# Optional dependency import for OpenCV with clear error message
try:
    cv2 = importlib.import_module("cv2")
except Exception as e:  # pragma: no cover - import resolution at runtime
    cv2 = None  # type: ignore[assignment]
    logger.debug("cv2 import failed at import time: %s", e)


def _preprocess_for_ocr(image_path: str) -> np.ndarray:
    """Basic preprocessing to clean noise and improve OCR.
    - Convert to grayscale
    - Resize up to improve readability
    - Adaptive threshold
    - Morphological opening to remove small artifacts
    """
    if settings.DEBUG:
        logger.info("OCR preprocess start: %s", image_path)
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for OCR preprocessing but is not installed.")
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Failed to read image for OCR")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # scale up small images
    h, w = gray.shape
    scale = 2 if max(h, w) < 1500 else 1
    if scale != 1:
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    # adaptive thresholding
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    # morphological opening to remove small noise
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    if settings.DEBUG:
        logger.info("OCR preprocess done: shape=%s", opened.shape)
    return opened


def ocr_image(image_path: str) -> Tuple[str, str]:
    """
    Run OCR on the image and return (raw_text, debug_text).
    debug_text can include engine info or be same as raw_text for now.
    """
    # Use preprocessing but keep PIL interface for compatibility
    if settings.DEBUG:
        logger.info("OCR detailed start: %s", image_path)
    proc = _preprocess_for_ocr(image_path)
    pil = Image.fromarray(proc)
    config = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ$€£.:,\\-/()&'\"#@%+*"  # noqa: E501
    text = pytesseract.image_to_string(pil, lang=settings.TESSERACT_LANG, config=config)
    return text, text


def ocr_image_detailed(image_path: str) -> Dict[str, Any]:
    """Return detailed OCR: text, lines, and word boxes with coordinates.
    Structure:
    {
      'text': str,
      'lines': [str, ...],
      'words': [ { 'text': str, 'left': int, 'top': int, 'width': int, 'height': int, 'line_id': (block, par, line) }, ...]
      'size': (width, height)
    }
    """
    proc = _preprocess_for_ocr(image_path)
    pil = Image.fromarray(proc)
    config = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ$€£.:,\\-/()&'\"#@%+*"  # noqa: E501
    data = pytesseract.image_to_data(pil, lang=settings.TESSERACT_LANG, output_type=pytesseract.Output.DICT, config=config)
    words: List[Dict[str, Any]] = []
    lines_map: Dict[tuple, List[int]] = {}
    all_lines_text: Dict[tuple, List[str]] = {}
    for i in range(len(data.get('text', []))):
        txt = (data['text'][i] or '').strip()
        if not txt:
            continue
        left, top, width, height = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        block, par, line = data['block_num'][i], data['par_num'][i], data['line_num'][i]
        lid = (block, par, line)
        words.append({
            'text': txt,
            'left': int(left),
            'top': int(top),
            'width': int(width),
            'height': int(height),
            'line_id': lid,
        })
        lines_map.setdefault(lid, []).append(len(words) - 1)
        all_lines_text.setdefault(lid, []).append(txt)
    # build ordered lines (by block, par, line)
    ordered_lids = sorted(lines_map.keys())
    lines: List[str] = [" ".join(all_lines_text[lid]) for lid in ordered_lids]
    text = "\n".join(lines)
    h, w = proc.shape
    if settings.DEBUG:
        logger.info("OCR detailed words=%d lines=%d size=%sx%s", len(words), len(lines), w, h)
    return {
        'text': text,
        'lines': lines,
        'words': words,
        'line_ids': ordered_lids,
        'size': (w, h),
    }


def draw_annotated_overlay(image_path: str, words: List[Dict[str, Any]], line_ids: List[tuple],
                           highlights: Dict[str, tuple]) -> np.ndarray:
    """Draw bounding boxes on the preprocessed image and highlight selected line_ids.
    highlights: dict like { 'date': lid, 'payee': lid, 'total': lid }
    Returns the BGR image with drawn rectangles.
    """
    proc = _preprocess_for_ocr(image_path)
    # Convert to BGR for colored drawing
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for drawing overlays but is not installed.")
    canvas = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
    # Build index of words by line id
    words_by_lid: Dict[tuple, List[Dict[str, Any]]] = {}
    for w in words:
        words_by_lid.setdefault(w['line_id'], []).append(w)
    # Draw all word boxes lightly
    for w in words:
        cv2.rectangle(
            canvas,
            (w['left'], w['top']),
            (w['left'] + w['width'], w['top'] + w['height']),
            (100, 100, 255),
            1,
        )
    # Highlight mappings
    colors = {
        'date': (0, 165, 255),   # Orange
        'payee': (0, 255, 0),    # Green
        'total': (0, 0, 255),    # Red
    }
    if highlights:
        for key, lid in highlights.items():
            if not lid:
                continue
            color = colors.get(key, (255, 0, 0))
            for w in words_by_lid.get(lid, []):
                cv2.rectangle(
                    canvas,
                    (w['left'], w['top']),
                    (w['left'] + w['width'], w['top'] + w['height']),
                    color,
                    2,
                )
    return canvas
