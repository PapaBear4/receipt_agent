from typing import Tuple, Dict, Any, List, Optional
from PIL import Image
import pytesseract
import importlib
import numpy as np

from app.config import settings
import logging
from app.services.image_preproc import crop_receipt, make_receipt_preview

# Module logger
logger = logging.getLogger(__name__)
if settings.DEBUG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# Optional dependency import for OpenCV with clear error message
cv2: Any  # typing hint for static analyzers
try:
    cv2 = importlib.import_module("cv2")
except Exception as e:  # pragma: no cover - import resolution at runtime
    cv2 = None  # type: ignore[assignment]
    logger.debug("cv2 import failed at import time: %s", e)


"""
MVP simplification: We removed automatic cropping/warping logic from this module.
Use app.services.image_preproc.crop_receipt/make_receipt_preview as placeholders
that simply return the original image for now. We'll revisit real detection later.
"""


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
    # MVP: assume user uploads pre-cropped images; pass-through from placeholder crop if desired
    try:
        img = make_receipt_preview(image_path)  # placeholder currently returns original image
    except Exception as e:
        logger.debug("Preview placeholder failed: %s", e)
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
    # Crop to text region to remove background margins
    try:
        inv = 255 - opened  # text becomes white
        nz = cv2.findNonZero(inv)
        if nz is not None:
            x, y, w_box, h_box = cv2.boundingRect(nz)
            H, W = opened.shape[:2]
            pad = max(10, int(0.02 * max(H, W)))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(W, x + w_box + pad)
            y1 = min(H, y + h_box + pad)
            cropped = opened[y0:y1, x0:x1]
        else:
            cropped = opened
    except Exception as e:
        logger.debug("Cropping to text region failed: %s", e)
        cropped = opened

    if settings.DEBUG:
        logger.info("OCR preprocess done: original=%s cropped=%s", opened.shape, cropped.shape)
    return cropped


def make_receipt_preview(image_path: str) -> np.ndarray:  # type: ignore[no-redef]
    """Re-export placeholder for preview to keep existing imports working."""
    return crop_receipt(image_path)


# crop_receipt is imported from app.services.image_preproc


def _light_preprocess_for_tesseract(img_bgr: np.ndarray) -> np.ndarray:
    """Light preprocessing intended not to destroy content.
    - Convert to grayscale
    - Gentle local contrast enhancement (CLAHE)
    - Avoid hard thresholding or resizing to keep coordinates stable
    Returns a single-channel uint8 image.
    """
    assert cv2 is not None, "cv2 is required"
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    try:
        clahe = cv2.createCLAHE(clipLimit=float(settings.OCR_CLAHE_CLIP), tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception as e:
        logger.debug("CLAHE failed, using plain grayscale: %s", e)
    return gray


def ocr_on_cropped_image(img_bgr: np.ndarray, debug_basename: Optional[str] = None) -> Dict[str, Any]:
    """Run OCR on a given (already cropped) image array with fallbacks.

    Tries multiple preprocessing variants and PSM modes, then returns the
    best result by word count. Also returns the processed image used, so
    overlays align perfectly.
    """
    assert cv2 is not None, "cv2 is required"

    # Raw-only short-circuit: send original image to Tesseract with no preprocessing/resizing
    if getattr(settings, "OCR_RAW_ONLY", False):
        # Prepare simple, non-destructive variants
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        variants = [("rgb", rgb), ("gray", gray), ("inv", inv)]
        # Build common config parts from settings
        oem = int(getattr(settings, "OCR_OEM", 1))
        whitelist = getattr(settings, "OCR_CHAR_WHITELIST", "") if getattr(settings, "OCR_USE_WHITELIST", True) else ""
        dict_flag = "-c load_system_dawg=0 load_freq_dawg=0" if getattr(settings, "OCR_DISABLE_DICTIONARY", False) else ""
        spaces_flag = "-c preserve_interword_spaces=1" if getattr(settings, "OCR_PRESERVE_SPACES", True) else ""
        dpi_flag = f"--dpi {int(getattr(settings, 'OCR_USER_DPI', 300))}"

        def run_raw(img_variant: np.ndarray, psm: int) -> Dict[str, Any]:
            config = f"--oem {oem} --psm {psm} {dpi_flag} {spaces_flag} {dict_flag}"
            if whitelist:
                config += f" -c tessedit_char_whitelist={whitelist}"
            pil_img = Image.fromarray(img_variant)
            data = pytesseract.image_to_data(pil_img, lang=settings.TESSERACT_LANG, output_type=pytesseract.Output.DICT, config=config)
            words: List[Dict[str, Any]] = []
            lines_map: Dict[tuple, List[int]] = {}
            all_lines_text: Dict[tuple, List[str]] = {}
            n = len(data.get('text', []))
            for i in range(n):
                txt = (data['text'][i] or '').strip()
                if not txt:
                    continue
                try:
                    conf = int(data.get('conf', ["-1"])[i])
                except Exception:
                    conf = -1
                if conf < settings.RECEIPT_CONF_THRESHOLD:
                    continue
                left, top, width, height = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                block, par, line = data['block_num'][i], data['par_num'][i], data['line_num'][i]
                lid = (block, par, line)
                words.append({'text': txt, 'left': int(left), 'top': int(top), 'width': int(width), 'height': int(height), 'line_id': lid})
                lines_map.setdefault(lid, []).append(len(words) - 1)
                all_lines_text.setdefault(lid, []).append(txt)
            ordered_lids = sorted(lines_map.keys())
            lines: List[str] = [" ".join(all_lines_text[lid]) for lid in ordered_lids]
            text = "\n".join(lines)
            # Preserve the variant’s size for coordinates
            if img_variant.ndim == 2:
                h, w = img_variant.shape
            else:
                h, w = img_variant.shape[:2]
            return {
                'text': text,
                'lines': lines,
                'words': words,
                'line_ids': ordered_lids,
                'size': (w, h),
                'proc_image': img_variant if img_variant.ndim == 3 else cv2.cvtColor(img_variant, cv2.COLOR_GRAY2BGR),
            }

        best: Optional[Dict[str, Any]] = None
        psms = [int(p.strip()) for p in str(getattr(settings, 'OCR_PSMS', '6,4,11,3,7')).split(',') if p.strip().isdigit()]
        for vname, vimg in variants:
            for psm in psms:
                res = run_raw(vimg, psm)
                if settings.DEBUG:
                    logger.debug("RAW OCR variant=%s psm=%d words=%d", vname, psm, len(res['words']))
                if best is None or len(res['words']) > len(best['words']):
                    best = res
        result = best if best is not None else run_raw(rgb, 6)
        if settings.DEBUG and debug_basename:
            try:
                out = settings.PROCESSED_DIR / f"ocr_input_used_{debug_basename}.png"
                cv2.imwrite(str(out), result['proc_image'])
                logger.info("Saved OCR input used (raw): %s", out)
            except Exception as e:
                logger.debug("Failed to save debug OCR input (raw): %s", e)
        return result

    def upscale_if_small(img: np.ndarray, target_max: int = 1800) -> np.ndarray:
        H, W = img.shape[:2]
        m = max(H, W)
        if m >= target_max:
            return img
        scale = float(target_max) / float(m)
        return cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_CUBIC)

    # Build preprocess variants
    gray = _light_preprocess_for_tesseract(img_bgr)
    gray = upscale_if_small(gray)
    variants = [("gray", gray)]
    # Optional Variant B: adaptive threshold + opening + optional median blur to reduce speckle
    if getattr(settings, "OCR_USE_THRESH", True):
        try:
            block = int(getattr(settings, "OCR_ADAPTIVE_BLOCK", 31))
            if block % 2 == 0:
                block += 1
            cval = int(getattr(settings, "OCR_ADAPTIVE_C", 10))
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, cval)
            kernel = np.ones((2, 2), np.uint8)
            opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
            mblur = int(getattr(settings, "OCR_MEDIAN_BLUR", 3))
            if mblur >= 3 and (mblur % 2 == 1):
                opened = cv2.medianBlur(opened, mblur)
            variants.append(("thresh", opened))
        except Exception as e:
            logger.debug("Adaptive threshold failed: %s", e)
    psms = [6, 4, 11, 7, 3]

    def run_ocr(img_single: np.ndarray, psm: int) -> Dict[str, Any]:
        pil_img = Image.fromarray(img_single)
        config = f"--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ$€£.:,\\-/()&@%+#*"
        data = pytesseract.image_to_data(pil_img, lang=settings.TESSERACT_LANG, output_type=pytesseract.Output.DICT, config=config)
        words: List[Dict[str, Any]] = []
        lines_map: Dict[tuple, List[int]] = {}
        all_lines_text: Dict[tuple, List[str]] = {}
        n = len(data.get('text', []))
        for i in range(n):
            txt = (data['text'][i] or '').strip()
            if not txt:
                continue
            try:
                conf = int(data.get('conf', ["-1"])[i])
            except Exception:
                conf = -1
            if conf < settings.RECEIPT_CONF_THRESHOLD:
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
        ordered_lids = sorted(lines_map.keys())
        lines: List[str] = [" ".join(all_lines_text[lid]) for lid in ordered_lids]
        text = "\n".join(lines)
        h, w = img_single.shape
        return {
            'text': text,
            'lines': lines,
            'words': words,
            'line_ids': ordered_lids,
            'size': (w, h),
        }

    best: Optional[Dict[str, Any]] = None
    best_meta = ("", 0)
    best_image: Optional[np.ndarray] = None
    for vname, vimg in variants:
        for psm in psms:
            result = run_ocr(vimg, psm)
            score = len(result['words'])
            if settings.DEBUG:
                logger.debug("OCR variant=%s psm=%d words=%d", vname, psm, score)
            if best is None or score > len(best['words']):
                best = result
                best_meta = (vname, psm)
                best_image = vimg

    if settings.DEBUG and debug_basename and best_image is not None:
        try:
            out = settings.PROCESSED_DIR / f"ocr_input_used_{debug_basename}.png"
            cv2.imwrite(str(out), best_image)
            logger.info("Saved OCR input used: %s (variant=%s, psm=%s)", out, best_meta[0], best_meta[1])
        except Exception as e:
            logger.debug("Failed to save debug OCR input: %s", e)

    if best is None:
        # Fallback to empty result
        h, w = gray.shape
        best = {'text': '', 'lines': [], 'words': [], 'line_ids': [], 'size': (w, h)}
        best_image = gray

    # Attach the image used for downstream overlay
    best['proc_image'] = best_image if best_image is not None else gray
    return best


def draw_overlay_on_image(base_img: np.ndarray, words: List[Dict[str, Any]], line_ids: List[tuple],
                          highlights: Dict[str, tuple]) -> np.ndarray:
    """Draw word boxes and highlight selected line_ids on a provided image.

    base_img can be grayscale or BGR; output is BGR. Coordinates must
    match the provided words (i.e., same image used for OCR).
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for drawing overlays but is not installed.")
    # Ensure color canvas
    if len(base_img.shape) == 2:
        canvas = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        canvas = base_img.copy()
    H, W = canvas.shape[:2]
    thickness_base = max(2, int(round(max(H, W) * 0.0025)))
    thickness_high = max(3, thickness_base * 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(2.0, max(H, W) / 1200.0))
    text_thickness = max(1, thickness_base)

    words_by_lid: Dict[tuple, List[Dict[str, Any]]] = {}
    for w in words:
        words_by_lid.setdefault(w['line_id'], []).append(w)

    for wd in words:
        cv2.rectangle(
            canvas,
            (wd['left'], wd['top']),
            (wd['left'] + wd['width'], wd['top'] + wd['height']),
            (80, 80, 80),
            thickness_base,
        )

    colors = {
        'date': (0, 165, 255),
        'payee': (0, 255, 0),
        'total': (0, 0, 255),
    }
    if highlights:
        for key, lid in highlights.items():
            if not lid:
                continue
            color = colors.get(key, (255, 0, 0))
            wlist = words_by_lid.get(lid, [])
            if not wlist:
                continue
            x1 = min(wd['left'] for wd in wlist)
            y1 = min(wd['top'] for wd in wlist)
            x2 = max(wd['left'] + wd['width'] for wd in wlist)
            y2 = max(wd['top'] + wd['height'] for wd in wlist)

            overlay_img = canvas.copy()
            alpha = 0.3
            for wd in wlist:
                cv2.rectangle(
                    overlay_img,
                    (wd['left'], wd['top']),
                    (wd['left'] + wd['width'], wd['top'] + wd['height']),
                    color,
                    -1,
                )
            cv2.addWeighted(overlay_img, alpha, canvas, 1 - alpha, 0, dst=canvas)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness_high)

            label = key.upper()
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            tx = x1
            ty = max(th + 6, y1 - 6)
            bg_x2 = min(W - 1, tx + tw + 8)
            bg_y1 = max(0, ty - th - 8)
            cv2.rectangle(canvas, (tx, bg_y1), (bg_x2, ty), color, -1)
            cv2.putText(canvas, label, (tx + 4, ty - 4), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
    return canvas


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
    config = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ$€£.:,\\-/()&@%+#*"  # noqa: E501
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
    config = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ$€£.:,\\-/()&@%+#*"  # noqa: E501
    data = pytesseract.image_to_data(pil, lang=settings.TESSERACT_LANG, output_type=pytesseract.Output.DICT, config=config)
    words: List[Dict[str, Any]] = []
    lines_map: Dict[tuple, List[int]] = {}
    all_lines_text: Dict[tuple, List[str]] = {}
    for i in range(len(data.get('text', []))):
        txt = (data['text'][i] or '').strip()
        if not txt:
            continue
        # Filter low-confidence detections to reduce noise outside the receipt
        try:
            conf = int(data.get('conf', ["-1"])[i])
        except Exception:
            conf = -1
        if conf < settings.RECEIPT_CONF_THRESHOLD:  # skip weak detections
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
    H, W = canvas.shape[:2]
    # Scale drawing parameters based on image size
    thickness_base = max(1, int(round(max(H, W) * 0.0015)))
    thickness_high = max(2, thickness_base * 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(2.0, max(H, W) / 1200.0))
    text_thickness = max(1, thickness_base)
    # Build index of words by line id
    words_by_lid: Dict[tuple, List[Dict[str, Any]]] = {}
    for w in words:
        words_by_lid.setdefault(w['line_id'], []).append(w)
    # Draw all word boxes lightly
    for wd in words:
        cv2.rectangle(
            canvas,
            (wd['left'], wd['top']),
            (wd['left'] + wd['width'], wd['top'] + wd['height']),
            (160, 160, 160),
            thickness_base,
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
            wlist = words_by_lid.get(lid, [])
            if not wlist:
                continue
            # Compute a bounding box that wraps the entire line
            x1 = min(wd['left'] for wd in wlist)
            y1 = min(wd['top'] for wd in wlist)
            x2 = max(wd['left'] + wd['width'] for wd in wlist)
            y2 = max(wd['top'] + wd['height'] for wd in wlist)

            # Semi-transparent fill over the words in this line for visibility
            overlay_img = canvas.copy()
            alpha = 0.3
            for wd in wlist:
                cv2.rectangle(
                    overlay_img,
                    (wd['left'], wd['top']),
                    (wd['left'] + wd['width'], wd['top'] + wd['height']),
                    color,
                    -1,
                )
            cv2.addWeighted(overlay_img, alpha, canvas, 1 - alpha, 0, dst=canvas)

            # Strong outline around the entire line
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness_high)

            # Label the line with the field name for clarity
            label = key.upper()
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            tx = x1
            ty = max(th + 6, y1 - 6)
            # Background box to keep text readable
            bg_x2 = min(W - 1, tx + tw + 8)
            bg_y1 = max(0, ty - th - 8)
            cv2.rectangle(canvas, (tx, bg_y1), (bg_x2, ty), color, -1)
            cv2.putText(canvas, label, (tx + 4, ty - 4), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
    return canvas
