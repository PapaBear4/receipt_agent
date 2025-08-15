import logging
import logging.config
from pathlib import Path
from app.config import settings
import uvicorn


def build_log_config(level_name: str, log_file: str) -> dict:
    """Return a logging config aligned with Uvicorn that also formats app logs and writes to a rotating file."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": "%(asctime)s | %(levelname)s | %(client_addr)s - \"%(request_line)s\" %(status_code)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
            "access": {
                "class": "logging.StreamHandler",
                "formatter": "access",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "default",
                "filename": log_file,
                "maxBytes": 5_000_000,
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "uvicorn": {"level": level_name, "handlers": ["default", "file"], "propagate": False},
            "uvicorn.error": {"level": level_name, "handlers": ["default", "file"], "propagate": False},
            "uvicorn.access": {"level": "INFO", "handlers": ["access", "file"], "propagate": False},
        },
        "root": {"level": level_name, "handlers": ["default", "file"]},
    }

if __name__ == "__main__":
    level_name = "DEBUG" if settings.DEBUG else "INFO"
    # Ensure logs directory exists under data
    logs_dir: Path = Path(settings.DATA_DIR) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "app.log"
    log_config = build_log_config(level_name, str(log_file))
    # Configure now to ensure any early logs use our formatter.
    logging.config.dictConfig(log_config)
    # Use import string so the app is imported after logging is configured.
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        log_config=log_config,
        log_level=level_name.lower(),
    )
