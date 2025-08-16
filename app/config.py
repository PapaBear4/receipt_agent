import os
from pathlib import Path


class Settings:
    # Networking
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

    # Paths
    _REPO_ROOT: Path = Path(__file__).resolve().parents[1] # Anchor default to repo root: <repo>/data
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(_REPO_ROOT / "data")))
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    # CSV_PATH resolution:
    # - If CSV_PATH env is set to an absolute path, use it as-is.
    # - If CSV_PATH env is set to a relative path, resolve it under DATA_DIR.
    # - If not set, default to DATA_DIR/ynab_receipts.csv.
    # Example:
    #   export CSV_PATH=custom.csv       -> <DATA_DIR>/custom.csv
    #   export CSV_PATH=/tmp/r.csv       -> /tmp/r.csv
    #   (unset)                          -> <DATA_DIR>/ynab_receipts.csv
    _CSV_ENV: str | None = os.getenv("CSV_PATH")
    if _CSV_ENV:
        _csv_path = Path(_CSV_ENV)
        CSV_PATH: Path = _csv_path if _csv_path.is_absolute() else DATA_DIR / _csv_path
    else:
        CSV_PATH: Path = DATA_DIR / "ynab_receipts.csv"

    # Database path (SQLite)
    _DB_ENV: str | None = os.getenv("DB_PATH")
    if _DB_ENV:
        _db_path = Path(_DB_ENV)
        DB_PATH: Path = _db_path if _db_path.is_absolute() else DATA_DIR / _db_path
    else:
        DB_PATH: Path = DATA_DIR / "receipts.db"

    # OCR
    TESSERACT_LANG: str = os.getenv("TESSERACT_LANG", "eng")

    # LLM (Ollama)
    OLLAMA_ENDPOINT: str = os.getenv("OLLAMA_ENDPOINT", "http://127.0.0.1:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_0")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "600")) # measured in seconds

    # LLM prompt sizing & content (env overridable)
    # If LLM_MAX_CHARS <= 0, do not truncate OCR text by characters.
    LLM_MAX_CHARS: int = int(os.getenv("LLM_MAX_CHARS", "0"))
    # Limit the number of indexed OCR lines included in the prompt (0 = no limit)
    LLM_MAX_LINES: int = int(os.getenv("LLM_MAX_LINES", "250"))
    # Include the full OCR text block in addition to indexed lines
    LLM_INCLUDE_FULL_TEXT: bool = os.getenv("LLM_INCLUDE_FULL_TEXT", "true").lower() in {"1", "true", "yes"}
    # When the OCR text is very long, prefer sending only indexed lines (drop full text)
    LLM_ONLY_INDEXED_WHEN_LONG: bool = os.getenv("LLM_ONLY_INDEXED_WHEN_LONG", "true").lower() in {"1", "true", "yes"}

    # Debugging
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in {"1", "true", "yes"}

    # Receipt detection and OCR tuning (env overridable)
    # Tip: export these before `python run.py` to experiment, e.g.:
    #   export RECEIPT_MIN_ASPECT=0.5 RECEIPT_MIN_HEIGHT_RATIO=0.35
    #   python run.py
    #
    # RECEIPT_TARGET_RESIZE_WIDTH: Resize width used during contour finding.
    #   Larger -> more detail (slower). Smaller -> faster but less precise. Typical 800–1400.
    RECEIPT_TARGET_RESIZE_WIDTH: int = int(os.getenv("RECEIPT_TARGET_RESIZE_WIDTH", "1000"))
    # RECEIPT_MIN_HEIGHT_RATIO: Minimum candidate quad height vs resized image height to accept as the receipt.
    #   Raise to avoid very short/flat regions (e.g., barcodes). Lower to accept smaller receipts. Typical 0.25–0.40.
    RECEIPT_MIN_HEIGHT_RATIO: float = float(os.getenv("RECEIPT_MIN_HEIGHT_RATIO", "0.30"))
    # RECEIPT_MIN_AREA_RATIO: Minimum candidate area fraction relative to resized image.
    #   Raise to avoid tiny candidates; lower for small/zoomed-out receipts. Typical 0.05–0.20.
    RECEIPT_MIN_AREA_RATIO: float = float(os.getenv("RECEIPT_MIN_AREA_RATIO", "0.10"))
    # RECEIPT_MIN_ASPECT: Minimum height/width ratio for the quad (filters extremely flat shapes like barcodes).
    #   Raise (e.g., 0.5) to be stricter against barcodes; lower (e.g., 0.3) for narrow/tall receipts.
    RECEIPT_MIN_ASPECT: float = float(os.getenv("RECEIPT_MIN_ASPECT", "0.40"))  # height/width
    # Fallback selection thresholds if a clean quad isn't found.
    # RECEIPT_FALLBACK_MIN_HEIGHT_RATIO: Accept bounding boxes only if tall enough.
    RECEIPT_FALLBACK_MIN_HEIGHT_RATIO: float = float(os.getenv("RECEIPT_FALLBACK_MIN_HEIGHT_RATIO", "0.25"))
    # RECEIPT_FALLBACK_MIN_ASPECT: Accept bounding boxes only if not too flat.
    RECEIPT_FALLBACK_MIN_ASPECT: float = float(os.getenv("RECEIPT_FALLBACK_MIN_ASPECT", "0.30"))
    # Padding around fallback crop (as fraction of image dims) and minimum absolute padding in pixels.
    RECEIPT_PAD_W: float = float(os.getenv("RECEIPT_PAD_W", "0.02"))  # as fraction of width
    RECEIPT_PAD_H: float = float(os.getenv("RECEIPT_PAD_H", "0.04"))  # as fraction of height
    RECEIPT_PAD_MIN: int = int(os.getenv("RECEIPT_PAD_MIN", "20"))
    # Final fallback: vertical center band if detection completely fails; this is the top/bottom margin fraction.
    RECEIPT_BAND_TOP: float = float(os.getenv("RECEIPT_BAND_TOP", "0.10"))  # fallback vertical band margin
    # Tesseract word confidence threshold; raise to suppress noisy words, lower to capture more.
    # MVP: be permissive to avoid missing text.
    RECEIPT_CONF_THRESHOLD: int = int(os.getenv("RECEIPT_CONF_THRESHOLD", "0"))

    # OCR preprocessing knobs (env overridable)
    # Toggle whether to try an adaptive-threshold variant (can create speckle on textured paper)
    OCR_USE_THRESH: bool = os.getenv("OCR_USE_THRESH", "false").lower() in {"1", "true", "yes"}
    # CLAHE contrast limiter (higher => stronger local contrast; too high can amplify paper grain)
    OCR_CLAHE_CLIP: float = float(os.getenv("OCR_CLAHE_CLIP", "2.0"))
    # Adaptive threshold parameters (must be odd block size)
    OCR_ADAPTIVE_BLOCK: int = int(os.getenv("OCR_ADAPTIVE_BLOCK", "31"))
    OCR_ADAPTIVE_C: int = int(os.getenv("OCR_ADAPTIVE_C", "10"))
    # Median blur kernel to remove salt-and-pepper speckle after threshold (odd, 0 to disable)
    OCR_MEDIAN_BLUR: int = int(os.getenv("OCR_MEDIAN_BLUR", "3"))
    # Raw-only mode: bypass all preprocessing and feed the original image to Tesseract
    OCR_RAW_ONLY: bool = os.getenv("OCR_RAW_ONLY", "true").lower() in {"1", "true", "yes"}
    # OCR engine/options
    # Comma-separated PSMs to try in order; we pick the result with the most words
    OCR_PSMS: str = os.getenv("OCR_PSMS", "11,6,4,3,7")
    # OCR engine mode: 1=LSTM only, 3=default; try 3 if results are too sparse
    OCR_OEM: int = int(os.getenv("OCR_OEM", "3"))
    # Whitelist handling: if false or empty, no whitelist constraint is passed
    OCR_USE_WHITELIST: bool = os.getenv("OCR_USE_WHITELIST", "false").lower() in {"1", "true", "yes"}
    OCR_CHAR_WHITELIST: str = os.getenv(
        "OCR_CHAR_WHITELIST",
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ$€£.:,\\-/()&@%+#*",
    )
    # Dictionary: disabling can reduce over-correction on receipts
    OCR_DISABLE_DICTIONARY: bool = os.getenv("OCR_DISABLE_DICTIONARY", "true").lower() in {"1", "true", "yes"}
    # Preserve spaces: improves layout fidelity in outputs
    OCR_PRESERVE_SPACES: bool = os.getenv("OCR_PRESERVE_SPACES", "true").lower() in {"1", "true", "yes"}
    # DPI hint for Tesseract; camera images benefit from a higher user-defined DPI
    OCR_USER_DPI: int = int(os.getenv("OCR_USER_DPI", "350"))


settings = Settings()

# Ensure directories exist at import time
for p in (settings.DATA_DIR, settings.UPLOADS_DIR, settings.PROCESSED_DIR):
    p.mkdir(parents=True, exist_ok=True)
