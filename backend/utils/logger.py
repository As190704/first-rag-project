"""
Centralized logging configuration for the RAG ingestion pipeline.

Provides a factory function to create named loggers with consistent
formatting across all modules.
"""

import logging
import sys
from pathlib import Path


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger instance.

    Args:
        name: The logger name, typically the module's __name__.

    Returns:
        A Logger instance writing to both stdout and a rotating log file.
    """
    logger = logging.getLogger(name)

    # Prevent adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Console handler ──────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    # ── File handler ─────────────────────────────────────────────────────────
    file_handler = logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger