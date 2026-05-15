"""Structured logging configuration for the insurance predictor API."""
from __future__ import annotations

import logging
import sys
from typing import Any


def configure_logging(level: str = "INFO") -> None:
    """Set up application-wide structured logging.

    Uses a consistent format across all modules: timestamp, level, logger name,
    and message. Call once at application startup.

    Args:
        level: Logging level string — one of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(numeric_level)
    if not root.handlers:
        root.addHandler(handler)
    else:
        root.handlers[0] = handler


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).

    Returns:
        Configured Logger instance.
    """
    return logging.getLogger(name)
