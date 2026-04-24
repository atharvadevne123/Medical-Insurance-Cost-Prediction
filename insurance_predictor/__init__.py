"""insurance_predictor — Medical Insurance Cost Prediction package."""
from .predictor import load_data, preprocess, train, predict, run

__all__ = ["load_data", "preprocess", "train", "predict", "run"]
__version__ = "1.0.0"
