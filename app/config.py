import os
from pathlib import Path


# Load env vars from local files without overriding existing variables
def _load_env_from_files() -> None:
    """Load key=value lines from optional local files into os.environ if not already set.
    Priority: repo/.env, ENV_FILE path, data/secrets.env.
    Comments (#) and blank lines are ignored. Does not override existing env vars.
    """
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / ".env",
        Path(os.getenv("ENV_FILE", "")) if os.getenv("ENV_FILE") else None,
        repo_root / "data" / "secrets.env",
    ]
    for p in [c for c in candidates if c]:
        try:
            if p.exists() and p.is_file():
                for line in p.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and (k not in os.environ):
                        os.environ[k] = v
        except Exception:
            # Best-effort; ignore parse errors
            pass


# Load local env before reading values into Settings
_load_env_from_files()


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
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    #OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_0")
    #OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "granite3.3:8b")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "3600")) # measured in seconds

    # LLM prompt sizing (env overridable)
    # Limit the number of indexed OCR lines included in the prompt (0 = no limit).
    LLM_MAX_LINES: int = int(os.getenv("LLM_MAX_LINES", "250"))

    # Debugging
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in {"1", "true", "yes"}

    # Persistence: automatically save OCR/LLM results to DB after background extraction
    # When true, the job pipeline writes receipts and line items immediately.
    # Manual review via /save will update the same rows (ON CONFLICT for receipts; clears and reinserts items).
    AUTO_SAVE_AFTER_EXTRACT: bool = os.getenv("AUTO_SAVE_AFTER_EXTRACT", "true").lower() in {"1", "true", "yes"}

    # YNAB API
    YNAB_API_BASE: str = os.getenv("YNAB_API_BASE", "https://api.ynab.com/v1")
    YNAB_TOKEN: str | None = os.getenv("YNAB_TOKEN")
    YNAB_BUDGET_ID: str | None = os.getenv("YNAB_BUDGET_ID")
    YNAB_DEFAULT_ACCOUNT_ID: str | None = os.getenv("YNAB_DEFAULT_ACCOUNT_ID")

    # OCR tuning (env overridable)
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
    OCR_PSMS: str = os.getenv("OCR_PSMS", "4,6,3,7")
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
    OCR_USER_DPI: int = int(os.getenv("OCR_USER_DPI", "300"))
    # Selection scoring weight: favor OCR variants that yield more distinct lines
    # Score = words + (OCR_SCORE_LINES_WEIGHT * lines). Increase to penalize merged lines.
    OCR_SCORE_LINES_WEIGHT: float = float(os.getenv("OCR_SCORE_LINES_WEIGHT", "0.8"))

    # Experimental: force OCR to operate on horizontal, full-width bands from top to bottom
    # When enabled, we segment the image into horizontal strips spanning the entire width,
    # then run Tesseract on each strip as a single line (PSM 7). Useful when default page
    # segmentation merges columns or skips narrow lines.
    OCR_FORCE_FULLWIDTH_LINES: bool = os.getenv("OCR_FORCE_FULLWIDTH_LINES", "true").lower() in {"1", "true", "yes"}
    # Minimum number of "ink" pixels per row, as a fraction of image width, to consider it part of a line
    OCR_FULLWIDTH_MIN_ROW_FRAC: float = float(os.getenv("OCR_FULLWIDTH_MIN_ROW_FRAC", "0.012"))
    # Optional: rows above this ink fraction are ignored (helps skip barcode stripes)
    OCR_FULLWIDTH_MAX_ROW_FRAC: float = float(os.getenv("OCR_FULLWIDTH_MAX_ROW_FRAC", "0.92"))
    # Smoothing window (rows) for the horizontal projection profile; odd number recommended
    # Default 9; override with OCR_FULLWIDTH_SMOOTH if provided
    OCR_FULLWIDTH_SMOOTH: int = int(os.getenv("OCR_FULLWIDTH_SMOOTH", "9"))
    # Merge short gaps (rows) between bands to prevent over-segmentation
    OCR_FULLWIDTH_MERGE_GAP: int = int(os.getenv("OCR_FULLWIDTH_MERGE_GAP", "1"))
    # Minimum band height (rows) to keep
    OCR_FULLWIDTH_MIN_HEIGHT: int = int(os.getenv("OCR_FULLWIDTH_MIN_HEIGHT", "8"))

    # Fallback: extra OCR pass focused on masked PAN/last4 (e.g., **** 1234, XXXX 1234)
    # Runs an additional quick pass with a strict whitelist "*Xx#0123456789/- " to help capture masked card lines
    # that Tesseract sometimes drops in general passes.
    OCR_PAN_FALLBACK: bool = os.getenv("OCR_PAN_FALLBACK", "true").lower() in {"1", "true", "yes"}

    # Post-OCR line clustering and ordering (env overridable)
    # Merge words into visual lines by Y and sort lines top-to-bottom, then words left-to-right.
    # y tolerance = max(OCR_Y_CLUSTER_MIN_PX, median_word_height * OCR_Y_CLUSTER_TOL_FRAC)
    OCR_Y_CLUSTER_TOL_FRAC: float = float(os.getenv("OCR_Y_CLUSTER_TOL_FRAC", "0.60"))
    OCR_Y_CLUSTER_MIN_PX: int = int(os.getenv("OCR_Y_CLUSTER_MIN_PX", "6"))

    # Item enrichment (optional web lookup)
    # Enable web enrichment to fetch better product metadata from vendor sites or search.
    ENRICH_ENABLED: bool = os.getenv("ENRICH_ENABLED", "false").lower() in {"1", "true", "yes"}
    # Provider: 'bing', 'serpapi', or 'none' (uses simple fetch with guessed URLs)
    ENRICH_PROVIDER: str = os.getenv("ENRICH_PROVIDER", "none")
    ENRICH_SEARCH_API_KEY: str | None = os.getenv("ENRICH_SEARCH_API_KEY")
    ENRICH_MAX_RESULTS: int = int(os.getenv("ENRICH_MAX_RESULTS", "3"))
    ENRICH_TIMEOUT: int = int(os.getenv("ENRICH_TIMEOUT", "12"))  # seconds
    ENRICH_RATE_LIMIT_PER_MIN: int = int(os.getenv("ENRICH_RATE_LIMIT_PER_MIN", "10"))
    ENRICH_USER_AGENT: str = os.getenv(
        "ENRICH_USER_AGENT",
        "receipt-agent/1.0 (+https://example.com; contact: dev@example.com)"
    )


settings = Settings()

# Ensure directories exist at import time
for p in (settings.DATA_DIR, settings.UPLOADS_DIR, settings.PROCESSED_DIR):
    p.mkdir(parents=True, exist_ok=True)
