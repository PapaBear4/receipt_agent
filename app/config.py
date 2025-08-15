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
    # CSV_PATH: if env set and relative, resolve against DATA_DIR; else default under DATA_DIR
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


settings = Settings()

# Ensure directories exist at import time
for p in (settings.DATA_DIR, settings.UPLOADS_DIR, settings.PROCESSED_DIR):
    p.mkdir(parents=True, exist_ok=True)
