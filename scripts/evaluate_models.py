"""Evaluate and compare all models on the insurance dataset."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import logging

import pandas as pd

from app.models import train_decision_tree, train_random_forest, train_ridge
from insurance_predictor.predictor import load_data, preprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluate_models")


def evaluate_all(csv_path: str | None = None) -> pd.DataFrame:
    """Train all models and return a comparison DataFrame.

    Args:
        csv_path: Optional path to a custom CSV dataset.

    Returns:
        DataFrame with one row per model and metric columns.
    """
    df = load_data(csv_path)
    X, Y = preprocess(df)
    rows = []
    trainers = {
        "Decision Tree": lambda: train_decision_tree(X, Y),
        "Random Forest": lambda: train_random_forest(X, Y, n_estimators=100),
        "Ridge Regression": lambda: train_ridge(X, Y),
    }
    for name, trainer in trainers.items():
        _, metrics = trainer()
        rows.append({"model": name, **metrics})
    results = pd.DataFrame(rows).set_index("model")
    results = results.sort_values("test_mae")
    return results


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    table = evaluate_all(csv_arg)
    print("\nModel Comparison:")
    print(table.to_string())
    best = table.index[0]
    print(f"\nBest model by test MAE: {best}")
