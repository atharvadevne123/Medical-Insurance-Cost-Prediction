"""insurance_predictor — Medical Insurance Cost Prediction package."""
from __future__ import annotations

from .predictor import feature_importance, load_data, predict, preprocess, run, train

__all__ = ["load_data", "preprocess", "train", "predict", "run", "feature_importance"]
__version__ = "1.0.0"
