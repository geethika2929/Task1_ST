import logging
import os
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler
from pathlib import Path

# Where logs live
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

APP_LOG_PATH = LOG_DIR / "app.log"
ERR_LOG_PATH = LOG_DIR / "error.log"

# We create one global logger for the whole app
LOGGER_NAME = "rag_app"


def get_logger() -> logging.Logger:
    """
    Returns a configured singleton logger.
    Safe to call from anywhere.
    """
    logger = logging.getLogger(LOGGER_NAME)

    # If it's already configured, just return it
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # capture everything; handlers will filter

    # --- Console handler (pretty, colored via rich) ---
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        markup=True,
    )
    console_handler.setLevel(logging.INFO)  # don't spam debug to console
    console_formatter = logging.Formatter(
        "%(message)s"
    )
    console_handler.setFormatter(console_formatter)

    # --- Rotating file handler for all logs (DEBUG+) ---
    file_handler = RotatingFileHandler(
        APP_LOG_PATH,
        maxBytes=5 * 1024 * 1024,  # ~5MB per file
        backupCount=5,            # keep last 5 files
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # --- Rotating file handler for ERROR+ only ---
    err_handler = RotatingFileHandler(
        ERR_LOG_PATH,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    err_handler.setLevel(logging.ERROR)
    err_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    err_handler.setFormatter(err_formatter)

    # Attach handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(err_handler)

    # Don't propagate to root logger (avoids double-printing)
    logger.propagate = False

    # Small startup note
    logger.debug("Logger initialized. APP_LOG_PATH=%s ERR_LOG_PATH=%s", APP_LOG_PATH, ERR_LOG_PATH)

    return logger
