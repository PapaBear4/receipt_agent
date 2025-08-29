from typing import Dict, Any, List, Optional
from PIL import Image
import pytesseract
import numpy as np

from app.config import settings
import logging
# (no longer importing image_preproc here; OCR expects an in-memory image array)

# Module logger
logger = logging.getLogger(__name__)
if settings.DEBUG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# OpenCV is required for preprocessing and overlays
try:
    import cv2  # type: ignore
except ImportError as e:  # pragma: no cover - import resolution at runtime
    raise ImportError("OpenCV (cv2) is required for OCR and overlays. Install with 'pip install opencv-python-headless'.") from e


"""
This module focuses on robust OCR from an already-provided image array.
"""

# Helper: cluster words into visual lines by Y position and sort lines top-to-bottom, then words left-to-right.
# Group OCR word boxes into lines by vertical proximity; returns lines, line_ids, updated words.
def _cluster_lines_by_y(words: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Given a flat list of word dicts with coords, build line clusters by Y.

    Returns a dict with keys:
    - lines: list[str] concatenated text per line
    - line_ids: list[tuple] identifiers (1,1,i)
    - words: updated words with 'line_id' rewritten to cluster IDs
    """
    if not words:
        return {"lines": [], "line_ids": [], "words": []}

    # Compute median height and per-word vertical centers
    heights = [max(1, int(w.get("height", 0) or 0)) for w in words if isinstance(w.get("height"), (int, float))]
    med_h = float(np.median(heights)) if heights else 12.0
    # Tolerances from settings
    try:
        frac = float(getattr(settings, "OCR_Y_CLUSTER_TOL_FRAC", 0.6))
    except Exception:
        frac = 0.6
    try:
        min_px = int(getattr(settings, "OCR_Y_CLUSTER_MIN_PX", 6))
    except Exception:
        min_px = 6
    y_tol = max(float(min_px), med_h * float(frac))

    # Prepare sortable list with index for stable updates
    idxs = list(range(len(words)))
    y_mids = [float(w["top"]) + float(w["height"]) / 2.0 for w in words]
    # Sort by y center primarily, then x (left)
    idxs.sort(key=lambda i: (y_mids[i], float(words[i]["left"])) )

    clusters: List[List[int]] = []  # lists of word indices
    cluster_ys: List[float] = []    # running mean y per cluster
    for i in idxs:
        y = y_mids[i]
        if not clusters:
            clusters.append([i])
            cluster_ys.append(y)
            continue
        # Compare to last cluster center
        if abs(y - cluster_ys[-1]) <= y_tol:
            clusters[-1].append(i)
            # update running mean for stability
            cluster_ys[-1] = (cluster_ys[-1] * (len(clusters[-1]) - 1) + y) / float(len(clusters[-1]))
        else:
            clusters.append([i])
            cluster_ys.append(y)

    # Build outputs: within each cluster, sort by x (left), then join
    new_words = words  # in-place update of line_id
    lines: List[str] = []
    lids: List[tuple] = []
    for li, inds in enumerate(clusters, start=1):
        inds_sorted = sorted(inds, key=lambda k: float(new_words[k]["left"]))
        lid = (1, 1, int(li))
        tokens: List[str] = []
        for k in inds_sorted:
            new_words[k]["line_id"] = lid
            tok = str(new_words[k].get("text", "")).strip()
            if tok:
                tokens.append(tok)
        lines.append(" ".join(tokens))
        lids.append(lid)

    return {"lines": lines, "line_ids": lids, "words": new_words}

# Light, coordinate-stable preprocessing for OCR: BGR->grayscale + optional CLAHE; no resize/thresh.
# Minimal preprocessing to improve OCR while preserving geometry and coords.
def _light_preprocess_for_tesseract(img_bgr: np.ndarray) -> np.ndarray:
    """Light preprocessing intended not to destroy content.
    - Convert to grayscale
    - Gentle local contrast enhancement (CLAHE)
    - Avoid hard thresholding or resizing to keep coordinates stable
    Returns a single-channel uint8 image.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    try:
        clahe = cv2.createCLAHE(clipLimit=float(settings.OCR_CLAHE_CLIP), tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception as e:
        logger.debug("CLAHE failed, using plain grayscale: %s", e)
    return gray

# Main OCR routine: tries preprocessing variants and PSMs, scores by words/lines; returns best result + proc_image for overlays.
# Run multi-variant Tesseract OCR with configurable options; returns text, lines, words, line_ids, size, proc_image.
def ocr_on_cropped_image(img_bgr: np.ndarray, debug_basename: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run OCR on a given (already cropped) image array with fallbacks.

    Tries multiple preprocessing variants and PSM modes, then returns the
    best result by word count. Also returns the processed image used, so
    overlays align perfectly.
    """

    # Resolve configuration values with optional per-call overrides
    def cfg(name: str, default: Any = None) -> Any:
        if overrides is not None and name in overrides:
            return overrides[name]
        return getattr(settings, name, default)

    DEBUG = bool(cfg("DEBUG", False))
    TESSERACT_LANG = str(cfg("TESSERACT_LANG", "eng"))
    RECEIPT_CONF_THRESHOLD = int(cfg("RECEIPT_CONF_THRESHOLD", 0))
    OCR_RAW_ONLY = bool(cfg("OCR_RAW_ONLY", False))
    OCR_OEM = int(cfg("OCR_OEM", 1))
    OCR_USE_WHITELIST = bool(cfg("OCR_USE_WHITELIST", True))
    # Avoid hard-coded escapes; fall back to settings default if not provided
    OCR_CHAR_WHITELIST = str(cfg("OCR_CHAR_WHITELIST", getattr(settings, "OCR_CHAR_WHITELIST", "")))
    OCR_DISABLE_DICTIONARY = bool(cfg("OCR_DISABLE_DICTIONARY", False))
    OCR_PRESERVE_SPACES = bool(cfg("OCR_PRESERVE_SPACES", True))
    OCR_USER_DPI = int(cfg("OCR_USER_DPI", 300))
    OCR_PSMS = [int(p.strip()) for p in str(cfg("OCR_PSMS", "6,4,11,3,7")).split(',') if p.strip().isdigit()]
    OCR_SCORE_LINES_WEIGHT = float(cfg("OCR_SCORE_LINES_WEIGHT", 0.5))
    OCR_USE_THRESH = bool(cfg("OCR_USE_THRESH", True))
    OCR_ADAPTIVE_BLOCK = int(cfg("OCR_ADAPTIVE_BLOCK", 31))
    OCR_ADAPTIVE_C = int(cfg("OCR_ADAPTIVE_C", 10))
    OCR_MEDIAN_BLUR = int(cfg("OCR_MEDIAN_BLUR", 3))
    # Full-width horizontal line mode
    OCR_FORCE_FULLWIDTH_LINES = bool(cfg("OCR_FORCE_FULLWIDTH_LINES", False))
    OCR_FULLWIDTH_MIN_ROW_FRAC = float(cfg("OCR_FULLWIDTH_MIN_ROW_FRAC", 0.012))
    OCR_FULLWIDTH_SMOOTH = int(cfg("OCR_FULLWIDTH_SMOOTH", 9))
    OCR_FULLWIDTH_MERGE_GAP = int(cfg("OCR_FULLWIDTH_MERGE_GAP", 3))
    OCR_FULLWIDTH_MIN_HEIGHT = int(cfg("OCR_FULLWIDTH_MIN_HEIGHT", 12))
    OCR_FULLWIDTH_MAX_ROW_FRAC = float(cfg("OCR_FULLWIDTH_MAX_ROW_FRAC", 0.92))

    # Raw-only short-circuit: send original image to Tesseract with no preprocessing/resizing
    if OCR_RAW_ONLY:
        # Prepare simple, non-destructive variants
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        variants = [("rgb", rgb), ("gray", gray), ("inv", inv)]
        # Build common config parts from settings/overrides
        oem = OCR_OEM
        whitelist = OCR_CHAR_WHITELIST if OCR_USE_WHITELIST else ""
        dict_flag = "-c load_system_dawg=0 load_freq_dawg=0" if OCR_DISABLE_DICTIONARY else ""
        spaces_flag = "-c preserve_interword_spaces=1" if OCR_PRESERVE_SPACES else ""
        dpi_flag = f"--dpi {OCR_USER_DPI}"

        def run_raw(img_variant: np.ndarray, psm: int) -> Dict[str, Any]:
            config = f"--oem {oem} --psm {psm} {dpi_flag} {spaces_flag} {dict_flag}"
            if whitelist:
                config += f" -c tessedit_char_whitelist={whitelist}"
            pil_img = Image.fromarray(img_variant)
            data = pytesseract.image_to_data(pil_img, lang=TESSERACT_LANG, output_type=pytesseract.Output.DICT, config=config)
            words: List[Dict[str, Any]] = []
            n = len(data.get('text', []))
            for i in range(n):
                txt = (data['text'][i] or '').strip()
                if not txt:
                    continue
                try:
                    conf = int(data.get('conf', ["-1"])[i])
                except Exception:
                    conf = -1
                if conf < RECEIPT_CONF_THRESHOLD:
                    continue
                left, top, width, height = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                # Temporarily assign source lid; will be overwritten by y-cluster
                words.append({'text': txt, 'left': int(left), 'top': int(top), 'width': int(width), 'height': int(height), 'line_id': (int(data['block_num'][i]), int(data['par_num'][i]), int(data['line_num'][i]))})
            grouped = _cluster_lines_by_y(words)
            lines: List[str] = grouped['lines']
            lids: List[tuple] = grouped['line_ids']
            text = "\n".join(lines)
            # Preserve the variant’s size for coordinates
            if img_variant.ndim == 2:
                h, w = img_variant.shape
            else:
                h, w = img_variant.shape[:2]
            return {
                'text': text,
                'lines': lines,
                'words': grouped['words'],
                'line_ids': lids,
                'size': (w, h),
                'proc_image': img_variant if img_variant.ndim == 3 else cv2.cvtColor(img_variant, cv2.COLOR_GRAY2BGR),
            }

        best: Optional[Dict[str, Any]] = None
        best_score: float = -1e9
        psms = OCR_PSMS
        for vname, vimg in variants:
            for psm in psms:
                res = run_raw(vimg, psm)
                score = float(len(res['words'])) + OCR_SCORE_LINES_WEIGHT * float(len(res['lines']))
                if DEBUG:
                    logger.debug("RAW OCR variant=%s psm=%d words=%d lines=%d score=%.2f", vname, psm, len(res['words']), len(res['lines']), score)
                if best is None or score > best_score:
                    best = res
                    best_score = score
        result = best if best is not None else run_raw(rgb, 6)
        # PAN fallback in raw-only path
        try:
            if bool(getattr(settings, 'OCR_PAN_FALLBACK', True)):
                pan_img = gray  # prefer grayscale for stability
                pan_whitelist = "*Xx#0123456789/- "
                oem = OCR_OEM
                dict_flag = "-c load_system_dawg=0 load_freq_dawg=0" if OCR_DISABLE_DICTIONARY else ""
                spaces_flag = "-c preserve_interword_spaces=1" if OCR_PRESERVE_SPACES else ""
                dpi_flag = f"--dpi {OCR_USER_DPI}"
                pan_psms = [7, 6, 11]
                pan_words: List[Dict[str, Any]] = []
                for psm in pan_psms:
                    try:
                        pan_config = f"--oem {oem} --psm {psm} {dpi_flag} {spaces_flag} {dict_flag} -c tessedit_char_whitelist={pan_whitelist}"
                        pil_img = Image.fromarray(pan_img)
                        data = pytesseract.image_to_data(pil_img, lang=TESSERACT_LANG, output_type=pytesseract.Output.DICT, config=pan_config)
                        n = len(data.get('text', []))
                        for i in range(n):
                            txt = (data['text'][i] or '').strip()
                            if not txt:
                                continue
                            left, top, width, height = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                            pan_words.append({'text': txt, 'left': int(left), 'top': int(top), 'width': int(width), 'height': int(height), 'line_id': (1,1,1)})
                    except Exception:
                        continue
                if pan_words:
                    grouped_pan = _cluster_lines_by_y(pan_words)
                    pan_lines: List[str] = grouped_pan['lines']
                    base_lines = result['lines'] if result else []
                    merged = list(base_lines)
                    for ln in pan_lines:
                        s = str(ln or '').strip()
                        if not s:
                            continue
                        if not any(s in (bl or '') or (bl or '') in s for bl in base_lines):
                            merged.append(s)
                    if result is not None:
                        result['lines'] = merged
                        result['text'] = "\n".join(merged)
        except Exception as e:
            if DEBUG:
                logger.debug("RAW-only PAN fallback failed: %s", e)
        if DEBUG and debug_basename:
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

    # If forcing full-width horizontal bands, do that specialized path first and return
    if OCR_FORCE_FULLWIDTH_LINES:
        def _segment_fullwidth_bands(gray_img: np.ndarray) -> List[tuple[int, int]]:
            # Compute horizontal projection of ink pixels, detect bands, merge small gaps.
            if gray_img.ndim != 2:
                g = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
            else:
                g = gray_img
            H, W = g.shape
            # Light binarization to detect ink; Otsu or adaptive
            try:
                _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            except Exception:
                bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10)
            ink = (255 - bw)  # text=white
            # A touch of vertical dilation helps connect characters within a line
            try:
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
                ink = cv2.dilate(ink, k, iterations=1)
            except Exception:
                pass
            row_counts = ink.sum(axis=1) / 255.0  # approx count of ink pixels per row
            # Smooth
            k = max(1, OCR_FULLWIDTH_SMOOTH)
            if k > 1:
                kernel = np.ones(k, dtype=np.float32) / float(k)
                row_counts = np.convolve(row_counts, kernel, mode='same')
            min_thresh = OCR_FULLWIDTH_MIN_ROW_FRAC * float(W)
            max_thresh = OCR_FULLWIDTH_MAX_ROW_FRAC * float(W)
            # Keep rows that are in the "texty" band: enough ink, but not near-solid (like barcodes)
            mask = (row_counts >= min_thresh) & (row_counts <= max_thresh)
            # Find contiguous True runs
            bands: List[tuple[int, int]] = []
            in_band = False
            start = 0
            for y in range(H):
                if mask[y] and not in_band:
                    in_band = True
                    start = y
                elif not mask[y] and in_band:
                    end = y - 1
                    bands.append((start, end))
                    in_band = False
            if in_band:
                bands.append((start, H - 1))
            # Merge small gaps
            merged: List[tuple[int, int]] = []
            for b in bands:
                if not merged:
                    merged.append(b)
                    continue
                prev_s, prev_e = merged[-1]
                if b[0] - prev_e - 1 <= OCR_FULLWIDTH_MERGE_GAP:
                    merged[-1] = (prev_s, b[1])
                else:
                    merged.append(b)
            # Enforce min height
            merged = [b for b in merged if (b[1] - b[0] + 1) >= OCR_FULLWIDTH_MIN_HEIGHT]
            return merged

        # Prepare grayscale for stable coordinates and upscale if small
        gray0 = _light_preprocess_for_tesseract(img_bgr)
        gray0 = cv2.resize(gray0, (gray0.shape[1], gray0.shape[0]), interpolation=cv2.INTER_AREA)
        gray = gray0
        bands = _segment_fullwidth_bands(gray)
        words: List[Dict[str, Any]] = []
        lines_text: List[str] = []
        line_ids: List[tuple] = []
        block = 1
        par = 1
        for i, (y0, y1) in enumerate(bands, start=1):
            strip = gray[y0:y1+1, :]
            pil_img = Image.fromarray(strip)
            # PSM 7: treat the image as a single text line
            whitelist = OCR_CHAR_WHITELIST if OCR_USE_WHITELIST else ""
            dict_flag = "-c load_system_dawg=0 load_freq_dawg=0" if OCR_DISABLE_DICTIONARY else ""
            spaces_flag = "-c preserve_interword_spaces=1" if OCR_PRESERVE_SPACES else ""
            dpi_flag = f"--dpi {OCR_USER_DPI}"
            config = f"--oem {OCR_OEM} --psm 7 {dpi_flag} {spaces_flag} {dict_flag}"
            if whitelist:
                config += f" -c tessedit_char_whitelist={whitelist}"
            data = pytesseract.image_to_data(pil_img, lang=TESSERACT_LANG, output_type=pytesseract.Output.DICT, config=config)
            n = len(data.get('text', []))
            lid = (block, par, i)
            line_tokens: List[str] = []
            row_words_idx: List[int] = []
            for j in range(n):
                txt = (data['text'][j] or '').strip()
                if not txt:
                    continue
                try:
                    conf = int(data.get('conf', ["-1"])[j])
                except Exception:
                    conf = -1
                if conf < RECEIPT_CONF_THRESHOLD:
                    continue
                left, top, width, height = data['left'][j], data['top'][j], data['width'][j], data['height'][j]
                # Map strip coords back to full image coords
                full_top = int(top) + int(y0)
                wdict = {
                    'text': txt,
                    'left': int(left),
                    'top': full_top,
                    'width': int(width),
                    'height': int(height),
                    'line_id': lid,
                }
                row_words_idx.append(len(words))
                words.append(wdict)
                line_tokens.append(txt)
            line_ids.append(lid)
            lines_text.append(" ".join(line_tokens))
        text = "\n".join(lines_text)
        result = {
            'text': text,
            'lines': lines_text,
            'words': words,
            'line_ids': line_ids,
            'size': (gray.shape[1], gray.shape[0]),
            'proc_image': cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        }
        # Full-width mode previously short-circuited before PAN fallback; run a quick masked-card pass here too
        try:
            if bool(getattr(settings, 'OCR_PAN_FALLBACK', True)):
                pan_img_base = gray
                # build alternative variants to help faint asterisks/digits
                pan_variants: List[tuple[str, np.ndarray]] = [('gray', pan_img_base)]
                try:
                    pan_variants.append(('inv', cv2.bitwise_not(pan_img_base)))
                    # light adaptive threshold
                    block = int(getattr(settings, 'OCR_ADAPTIVE_BLOCK', 31))
                    if block % 2 == 0:
                        block += 1
                    cval = int(getattr(settings, 'OCR_ADAPTIVE_C', 10))
                    th = cv2.adaptiveThreshold(pan_img_base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, cval)
                    pan_variants.append(('thresh', th))
                except Exception:
                    pass
                pan_whitelist = "*Xx#0123456789/- "
                dict_flag = "-c load_system_dawg=0 load_freq_dawg=0" if bool(getattr(settings, 'OCR_DISABLE_DICTIONARY', False)) else ""
                spaces_flag = "-c preserve_interword_spaces=1" if bool(getattr(settings, 'OCR_PRESERVE_SPACES', True)) else ""
                dpi_flag = f"--dpi {int(getattr(settings, 'OCR_USER_DPI', 300))}"
                pan_psms = [7, 6, 11]
                pan_words: List[Dict[str, Any]] = []
                for _, pv in pan_variants:
                    for psm in pan_psms:
                        try:
                            pan_config = f"--oem {int(getattr(settings, 'OCR_OEM', 1))} --psm {psm} {dpi_flag} {spaces_flag} {dict_flag} -c tessedit_char_whitelist={pan_whitelist}"
                            pil_img = Image.fromarray(pv)
                            data = pytesseract.image_to_data(pil_img, lang=str(getattr(settings, 'TESSERACT_LANG', 'eng')), output_type=pytesseract.Output.DICT, config=pan_config)
                            n = len(data.get('text', []))
                            for j in range(n):
                                txt = (data['text'][j] or '').strip()
                                if not txt:
                                    continue
                                pan_words.append({'text': txt, 'left': int(data['left'][j]), 'top': int(data['top'][j]), 'width': int(data['width'][j]), 'height': int(data['height'][j]), 'line_id': (1,1,1)})
                        except Exception:
                            continue
                if pan_words:
                    grouped_pan = _cluster_lines_by_y(pan_words)
                    pan_lines: List[str] = grouped_pan['lines']
                    # Keep only plausible masked-card lines to avoid noise
                    import re as _re
                    MASKED_RE = _re.compile(r"(?:\*{2,}|X{2,}|x{2,}|#){2,}.*?\b\d{4}\b")
                    filtered = [ln for ln in pan_lines if MASKED_RE.search(str(ln or ''))]
                    if filtered:
                        base_lines = result['lines']
                        merged = list(base_lines)
                        for ln in filtered:
                            s = str(ln or '').strip()
                            if not s:
                                continue
                            if not any(s in (bl or '') or (bl or '') in s for bl in base_lines):
                                merged.append(s)
                        result['lines'] = merged
                        result['text'] = "\n".join(merged)
        except Exception:
            pass
        if DEBUG and debug_basename is not None:
            try:
                out = settings.PROCESSED_DIR / f"ocr_input_used_{debug_basename}.png"
                cv2.imwrite(str(out), gray)
            except Exception:
                pass
        return result

    # Build preprocess variants
    gray = _light_preprocess_for_tesseract(img_bgr)
    gray = upscale_if_small(gray)
    variants = [("gray", gray)]
    # Optional Variant B: adaptive threshold + opening + optional median blur to reduce speckle
    if OCR_USE_THRESH:
        try:
            block = int(OCR_ADAPTIVE_BLOCK)
            if block % 2 == 0:
                block += 1
            cval = int(OCR_ADAPTIVE_C)
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, cval)
            kernel = np.ones((2, 2), np.uint8)
            opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
            mblur = int(OCR_MEDIAN_BLUR)
            if mblur >= 3 and (mblur % 2 == 1):
                opened = cv2.medianBlur(opened, mblur)
            variants.append(("thresh", opened))
        except Exception as e:
            logger.debug("Adaptive threshold failed: %s", e)
    psms = OCR_PSMS if OCR_PSMS else [6, 4, 11, 7, 3]

    def run_ocr(img_single: np.ndarray, psm: int) -> Dict[str, Any]:
        pil_img = Image.fromarray(img_single)
        # Build config honoring lab/settings controls
        oem = OCR_OEM
        whitelist = OCR_CHAR_WHITELIST if OCR_USE_WHITELIST else ""
        dict_flag = "-c load_system_dawg=0 load_freq_dawg=0" if OCR_DISABLE_DICTIONARY else ""
        spaces_flag = "-c preserve_interword_spaces=1" if OCR_PRESERVE_SPACES else ""
        dpi_flag = f"--dpi {OCR_USER_DPI}"
        config = f"--oem {oem} --psm {psm} {dpi_flag} {spaces_flag} {dict_flag}"
        if whitelist:
            config += f" -c tessedit_char_whitelist={whitelist}"
        data = pytesseract.image_to_data(pil_img, lang=TESSERACT_LANG, output_type=pytesseract.Output.DICT, config=config)
        words: List[Dict[str, Any]] = []
        n = len(data.get('text', []))
        for i in range(n):
            txt = (data['text'][i] or '').strip()
            if not txt:
                continue
            try:
                conf = int(data.get('conf', ["-1"])[i])
            except Exception:
                conf = -1
            if conf < RECEIPT_CONF_THRESHOLD:
                continue
            left, top, width, height = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            # Temporarily assign source lid; will be overwritten by y-cluster
            words.append({
                'text': txt,
                'left': int(left),
                'top': int(top),
                'width': int(width),
                'height': int(height),
                'line_id': (int(data['block_num'][i]), int(data['par_num'][i]), int(data['line_num'][i])),
            })
        grouped = _cluster_lines_by_y(words)
        lines: List[str] = grouped['lines']
        text = "\n".join(lines)
        lids: List[tuple] = grouped['line_ids']
        h, w = img_single.shape
        return {
            'text': text,
            'lines': lines,
            'words': grouped['words'],
            'line_ids': lids,
            'size': (w, h),
        }

    best: Optional[Dict[str, Any]] = None
    best_meta = ("", 0)
    best_image: Optional[np.ndarray] = None
    best_score: float = -1e9
    for vname, vimg in variants:
        for psm in psms:
            result = run_ocr(vimg, psm)
            score = float(len(result['words'])) + OCR_SCORE_LINES_WEIGHT * float(len(result['lines']))
            if DEBUG:
                logger.debug("OCR variant=%s psm=%d words=%d lines=%d score=%.2f", vname, psm, len(result['words']), len(result['lines']), score)
            if best is None or score > best_score:
                best = result
                best_meta = (vname, psm)
                best_image = vimg
                best_score = score

    if DEBUG and debug_basename and best_image is not None:
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

    # Optional: masked PAN fallback pass to capture lines like "**** 1234"
    try:
        if bool(getattr(settings, 'OCR_PAN_FALLBACK', True)):
            pan_img_base = gray  # use grayscale variant for stability
            # Try multiple variants to improve faint symbols
            pan_variants: List[tuple[str, np.ndarray]] = [('gray', pan_img_base)]
            try:
                pan_variants.append(('inv', cv2.bitwise_not(pan_img_base)))
                block = int(OCR_ADAPTIVE_BLOCK)
                if block % 2 == 0:
                    block += 1
                cval = int(OCR_ADAPTIVE_C)
                th = cv2.adaptiveThreshold(pan_img_base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, cval)
                pan_variants.append(('thresh', th))
            except Exception:
                pass
            # Build a strict whitelist focused on masked PAN tokens
            pan_whitelist = "*Xx#0123456789/- "
            oem = OCR_OEM
            dict_flag = "-c load_system_dawg=0 load_freq_dawg=0" if OCR_DISABLE_DICTIONARY else ""
            spaces_flag = "-c preserve_interword_spaces=1" if OCR_PRESERVE_SPACES else ""
            dpi_flag = f"--dpi {OCR_USER_DPI}"
            # Try PSMs that treat short text lines well
            pan_psms = [7, 6, 11]
            pan_words: List[Dict[str, Any]] = []
            for _, pv in pan_variants:
                for psm in pan_psms:
                    try:
                        pan_config = f"--oem {oem} --psm {psm} {dpi_flag} {spaces_flag} {dict_flag} -c tessedit_char_whitelist={pan_whitelist}"
                        pil_img = Image.fromarray(pv)
                        data = pytesseract.image_to_data(pil_img, lang=TESSERACT_LANG, output_type=pytesseract.Output.DICT, config=pan_config)
                        n = len(data.get('text', []))
                        for i in range(n):
                            txt = (data['text'][i] or '').strip()
                            if not txt:
                                continue
                            # keep low confidence too; masked lines can be faint
                            pan_words.append({'text': txt, 'left': int(data['left'][i]), 'top': int(data['top'][i]), 'width': int(data['width'][i]), 'height': int(data['height'][i]), 'line_id': (1,1,1)})
                    except Exception as e:
                        if DEBUG:
                            logger.debug("PAN fallback PSM %s failed: %s", psm, e)
                        continue
            if pan_words:
                grouped_pan = _cluster_lines_by_y(pan_words)
                pan_lines: List[str] = grouped_pan['lines']
                # Merge only plausible masked card lines to avoid noise
                import re as _re
                MASKED_RE = _re.compile(r"(?:\*{2,}|X{2,}|x{2,}|#){2,}.*?\b\d{4}\b")
                filtered = [ln for ln in pan_lines if MASKED_RE.search(str(ln or ''))]
                if filtered:
                    base_lines = best['lines'] if best else []
                    merged = list(base_lines)
                    for ln in filtered:
                        s = str(ln or '').strip()
                        if not s:
                            continue
                        # Deduplicate by simple containment; keep if not present in any existing line
                        if not any(s in (bl or '') or (bl or '') in s for bl in base_lines):
                            merged.append(s)
                    if best is not None:
                        best['lines'] = merged
                        best['text'] = "\n".join(merged)

            # ROI rescan just below payment method hints (DEBIT/CREDIT/PURCHASE)
            try:
                import re as _re
                H, W = gray.shape[:2]
                # Map words to line_ids and texts to locate candidate line positions
                words = best.get('words', []) if isinstance(best, dict) else []
                by_lid: Dict[tuple, List[Dict[str, Any]]] = {}
                for w in words:
                    lid_key = w.get('line_id')
                    if isinstance(lid_key, tuple):
                        by_lid.setdefault(lid_key, []).append(w)
                # Build line text map using best lines order
                line_texts = list(best.get('lines', []))
                # Keyword set
                KEYS = ('DEBIT', 'CREDIT', 'PURCHASE', 'CARD')
                # Find lids whose concatenated words contain any of these keys
                cand_lids: List[tuple] = []
                # Approximate by scanning unique lids in order of top y
                lids_sorted: List[tuple] = []
                for lid, lst in by_lid.items():
                    if not lst:
                        continue
                    lids_sorted.append((min(w['top'] for w in lst), lid))
                lids_sorted.sort(key=lambda t: t[0])
                for _, lid in lids_sorted:
                    toks = [str(w.get('text', '')).strip() for w in by_lid.get(lid, [])]
                    s = " ".join([t for t in toks if t])
                    up = s.upper()
                    if any(k in up for k in KEYS):
                        cand_lids.append(lid)
                if cand_lids:
                    # compute median word height for band sizing
                    heights = [max(1, int(w.get('height', 1))) for w in words]
                    med_h = float(np.median(heights)) if heights else 16.0
                    band_h = int(max(18.0, med_h * 2.5))
                    pan_whitelist = "*Xx#0123456789/- "
                    dict_flag = "-c load_system_dawg=0 load_freq_dawg=0" if OCR_DISABLE_DICTIONARY else ""
                    spaces_flag = "-c preserve_interword_spaces=1" if OCR_PRESERVE_SPACES else ""
                    dpi_flag = f"--dpi {OCR_USER_DPI}"
                    psms = [7]
                    MASKED_RE = _re.compile(r"(?:\\*{2,}|X{2,}|x{2,}|#){2,}.*?\\b\\d{4}\\b")
                    found_lines: List[str] = []
                    for lid in cand_lids[:3]:  # limit to first few
                        lst = by_lid.get(lid, [])
                        if not lst:
                            continue
                        top = min(w['top'] for w in lst)
                        y0 = int(min(H-1, max(0, top + int(med_h*0.8))))
                        y1 = int(min(H, y0 + band_h))
                        roi = gray[y0:y1, :]
                        # variants
                        variants = [('gray', roi)]
                        try:
                            variants.append(('inv', cv2.bitwise_not(roi)))
                            block = int(OCR_ADAPTIVE_BLOCK)
                            if block % 2 == 0: block += 1
                            th = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, int(OCR_ADAPTIVE_C))
                            variants.append(('thresh', th))
                        except Exception:
                            pass
                        for _, vv in variants:
                            for psm in psms:
                                try:
                                    cfg_str = f"--oem {OCR_OEM} --psm {psm} {dpi_flag} {spaces_flag} {dict_flag} -c tessedit_char_whitelist={pan_whitelist}"
                                    data = pytesseract.image_to_data(Image.fromarray(vv), lang=TESSERACT_LANG, output_type=pytesseract.Output.DICT, config=cfg_str)
                                    n = len(data.get('text', []))
                                    # Join tokens left-to-right in this ROI
                                    toks = []
                                    for i in range(n):
                                        t = (data['text'][i] or '').strip()
                                        if t:
                                            toks.append((int(data['left'][i]), t))
                                    toks.sort(key=lambda t: t[0])
                                    s = " ".join(t for _, t in toks)
                                    if s and (MASKED_RE.search(s) or _re.search(r"\\b\\d{4}\\b", s)):
                                        found_lines.append(s)
                                except Exception:
                                    continue
                    if found_lines:
                        base_lines = best['lines'] if best else []
                        merged = list(base_lines)
                        for s in found_lines:
                            s = s.strip()
                            if not any(s in (bl or '') or (bl or '') in s for bl in base_lines):
                                merged.append(s)
                        if best is not None:
                            best['lines'] = merged
                            best['text'] = "\\n".join(merged)
            except Exception:
                pass
    except Exception as e:
        if DEBUG:
            logger.debug("PAN fallback failed: %s", e)

    # Final attempt: ROI-based masked PAN search near payment keywords (DEBIT/CREDIT/PURCHASE)
    try:
        import re as _re
        MASKED_RE = _re.compile(r"(?:\*{2,}|X{2,}|x{2,}|#){2,}.*?\b\d{4}\b")
        already = any(MASKED_RE.search(str(ln or '')) for ln in (best.get('lines') or []))
        if not already and (best_image is not None):
            # Build quick lookup of words by line_id
            words = best.get('words') or []
            words_by_lid: Dict[tuple, List[Dict[str, Any]]] = {}
            for wdict in words:
                lid = tuple(wdict.get('line_id') or ())
                if lid:
                    words_by_lid.setdefault(lid, []).append(wdict)
            lids = best.get('line_ids') or []
            lines = best.get('lines') or []
            H, W = (best_image.shape[:2] if best_image.ndim == 3 else best_image.shape)
            # ensure grayscale ROI base
            if best_image.ndim == 3:
                roi_base = cv2.cvtColor(best_image, cv2.COLOR_BGR2GRAY)
            else:
                roi_base = best_image.copy()
            KEY_UP = ("DEBIT", "CREDIT", "PURCHASE", "US DEBIT", "US CREDIT", "CARD")

            def _crop_safe(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
                x1 = max(0, min(W-1, int(x1)))
                y1 = max(0, min(H-1, int(y1)))
                x2 = max(0, min(W, int(x2)))
                y2 = max(0, min(H, int(y2)))
                if x2 <= x1 or y2 <= y1:
                    return roi_base[0:0, 0:0]
                return roi_base[y1:y2, x1:x2]

            def _run_pan_ocr(img_single: np.ndarray) -> List[str]:
                if img_single.size == 0:
                    return []
                vars: List[np.ndarray] = []
                vars.append(img_single)
                try:
                    vars.append(cv2.bitwise_not(img_single))
                except Exception:
                    pass
                try:
                    block = int(OCR_ADAPTIVE_BLOCK)
                    if block % 2 == 0:
                        block += 1
                    cval = int(OCR_ADAPTIVE_C)
                    th = cv2.adaptiveThreshold(img_single, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, cval)
                    vars.append(th)
                except Exception:
                    pass
                wl = "*Xx#0123456789/- "
                oem = OCR_OEM
                dict_flag = "-c load_system_dawg=0 load_freq_dawg=0" if OCR_DISABLE_DICTIONARY else ""
                spaces_flag = "-c preserve_interword_spaces=1" if OCR_PRESERVE_SPACES else ""
                dpi_flag = f"--dpi {OCR_USER_DPI}"
                psms = [7, 6, 11]
                out_lines: List[str] = []
                for v in vars:
                    pil = Image.fromarray(v)
                    for psm in psms:
                        try:
                            cfg_str = f"--oem {oem} --psm {psm} {dpi_flag} {spaces_flag} {dict_flag} -c tessedit_char_whitelist={wl}"
                            data = pytesseract.image_to_data(pil, lang=TESSERACT_LANG, output_type=pytesseract.Output.DICT, config=cfg_str)
                            n = len(data.get('text', []))
                            toks: List[str] = []
                            for i in range(n):
                                t = (data['text'][i] or '').strip()
                                if t:
                                    toks.append(t)
                            if toks:
                                out_lines.append(" ".join(toks))
                        except Exception:
                            continue
                # Keep only masked candidates
                return [s for s in out_lines if MASKED_RE.search(str(s))]

            found: List[str] = []
            last_roiB = None  # track for digits-only fallback
            for i, lid in enumerate(lids):
                s = (lines[i] or "") if i < len(lines) else ""
                up = s.upper()
                if not any(k in up for k in KEY_UP):
                    continue
                wlist = words_by_lid.get(tuple(lid), [])
                if not wlist:
                    continue
                x1 = min(w['left'] for w in wlist)
                y1 = min(w['top'] for w in wlist)
                x2 = max(w['left'] + w['width'] for w in wlist)
                y2 = max(w['top'] + w['height'] for w in wlist)
                h = max(6, y2 - y1)
                # ROI A: to the right of the keyword line
                roiA = _crop_safe(x2 + 4, y1 - int(0.4*h), min(W, x2 + int(0.55*W)), y2 + int(0.6*h))
                found += _run_pan_ocr(roiA)
                # ROI B: next line below, full right half
                roiB = _crop_safe(int(W*0.35), y2 + 2, W, min(H, y2 + int(2.2*h)))
                last_roiB = roiB
                found += _run_pan_ocr(roiB)
            # Merge into best if any new masked lines detected
            if not found:
                # Secondary: digits-only sweep in same ROIs to recover last4 even when mask glyphs fail
                def _digits_only(img_single: np.ndarray) -> str:
                    if img_single.size == 0:
                        return ""
                    wl = "0123456789"
                    oem = OCR_OEM
                    dict_flag = "-c load_system_dawg=0 load_freq_dawg=0" if OCR_DISABLE_DICTIONARY else ""
                    spaces_flag = "-c preserve_interword_spaces=1" if OCR_PRESERVE_SPACES else ""
                    dpi_flag = f"--dpi {OCR_USER_DPI}"
                    psms = [7, 6, 11]
                    vars: List[np.ndarray] = [img_single]
                    try:
                        vars.append(cv2.bitwise_not(img_single))
                    except Exception:
                        pass
                    try:
                        block = int(OCR_ADAPTIVE_BLOCK)
                        if block % 2 == 0:
                            block += 1
                        cval = int(OCR_ADAPTIVE_C)
                        th = cv2.adaptiveThreshold(img_single, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, cval)
                        vars.append(th)
                    except Exception:
                        pass
                    best_seq = ""
                    import re as _re2
                    re4 = _re2.compile(r"\d{4,}")
                    for v in vars:
                        pil = Image.fromarray(v)
                        for psm in psms:
                            try:
                                cfg_str = f"--oem {oem} --psm {psm} {dpi_flag} {spaces_flag} {dict_flag} -c tessedit_char_whitelist={wl}"
                                data = pytesseract.image_to_data(pil, lang=TESSERACT_LANG, output_type=pytesseract.Output.DICT, config=cfg_str)
                                toks = [(data['text'][k] or '').strip() for k in range(len(data.get('text', []))) if (data['text'][k] or '').strip()]
                                s = "".join(toks)
                                m = re4.search(s)
                                if m:
                                    seq = m.group(0)
                                    if len(seq) >= 4:
                                        best_seq = seq[-4:]
                                        return best_seq
                            except Exception:
                                continue
                    return best_seq
                # try digits-only around the first matched keyword ROI we computed last
                # reuse last roiB if available; otherwise attempt the lower-right quadrant near payment area
                candidate = _digits_only(last_roiB) if last_roiB is not None else ""
                if candidate and len(candidate) == 4:
                    found.append(f"**** {candidate}")

            if found:
                base_lines = best['lines']
                merged = list(base_lines)
                for ln in found:
                    s = str(ln or '').strip()
                    if s and not any(s in (bl or '') or (bl or '') in s for bl in base_lines):
                        merged.append(s)
                best['lines'] = merged
                best['text'] = "\n".join(merged)
    except Exception as e:
        if DEBUG:
            logger.debug("ROI PAN search failed: %s", e)

    # Attach the image used for downstream overlay
    best['proc_image'] = best_image if best_image is not None else gray
    return best

# Draw utility: renders all word boxes and optionally highlights specific line_ids (date/payee/total).
# Render overlay rectangles on the provided image; optionally highlight date/payee/total regions.
def draw_overlay_on_image(base_img: np.ndarray, words: List[Dict[str, Any]], line_ids: List[tuple],
                          highlights: Dict[str, tuple]) -> np.ndarray:
    """Draw word boxes and highlight selected line_ids on a provided image.

    base_img can be grayscale or BGR; output is BGR. Coordinates must
    match the provided words (i.e., same image used for OCR).
    """
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

    


