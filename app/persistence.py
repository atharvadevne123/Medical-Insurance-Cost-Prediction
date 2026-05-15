"""Model persistence: save and load trained models using joblib."""
from __future__ import annotations

import logging
import pathlib
from typing import Any

import joblib

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = pathlib.Path("models")


def save_model(model: Any, path: str | pathlib.Path) -> pathlib.Path:
    """Persist a fitted model to disk.

    Args:
        model: Any fitted sklearn-compatible model.
        path: Destination file path (will create parent directories).

    Returns:
        Resolved path where the model was written.
    """
    dest = pathlib.Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(model, dest)
        logger.info("Model saved to %s", dest)
    except Exception as exc:
        logger.error("Failed to save model to %s: %s", dest, exc)
        raise
    return dest


def load_model(path: str | pathlib.Path) -> Any:
    """Load a model that was saved with save_model.

    Args:
        path: Path to the serialised model file.

    Returns:
        The de-serialised model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    src = pathlib.Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Model file not found: {src}")
    try:
        model = joblib.load(src)
        logger.info("Model loaded from %s", src)
        return model
    except Exception as exc:
        logger.error("Failed to load model from %s: %s", src, exc)
        raise


def model_exists(path: str | pathlib.Path) -> bool:
    """Return True if a saved model file exists at path."""
    return pathlib.Path(path).exists()
