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

    # OCR
    TESSERACT_LANG: str = os.getenv("TESSERACT_LANG", "eng")

    # LLM (Ollama)
    OLLAMA_ENDPOINT: str = os.getenv("OLLAMA_ENDPOINT", "http://127.0.0.1:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b_q4")

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
    RECEIPT_CONF_THRESHOLD: int = int(os.getenv("RECEIPT_CONF_THRESHOLD", "10"))


settings = Settings()

# Ensure directories exist at import time
for p in (settings.DATA_DIR, settings.UPLOADS_DIR, settings.PROCESSED_DIR):
    p.mkdir(parents=True, exist_ok=True)
