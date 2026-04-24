"""Medical insurance cost predictor — extracted from the original notebook."""
from __future__ import annotations
import os
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

_DATA = pathlib.Path(__file__).parent / "data" / "insurance.csv"


def load_data(csv_path: str | None = None) -> pd.DataFrame:
    """Load the insurance dataset. Defaults to the bundled CSV."""
    path = csv_path or str(_DATA)
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame):
    """Return (X, Y) ready for training."""
    Y = df[["charges"]]
    X = df.drop(columns=["charges"])
    cat = [c for c in X.columns if X[c].dtype == object]
    con = [c for c in X.columns if X[c].dtype != object]
    ss = StandardScaler()
    X_con = pd.DataFrame(ss.fit_transform(X[con]), columns=con)
    X_cat = pd.get_dummies(X[cat])
    return X_con.join(X_cat), Y


def train(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    max_depth: int = 6,
    min_samples_split: int = 9,
    test_size: float = 0.2,
    random_state: int = 20,
) -> tuple[DecisionTreeRegressor, dict]:
    """Train a DecisionTreeRegressor and return (model, metrics)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    model = DecisionTreeRegressor(
        criterion="absolute_error",
        random_state=21,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
    )
    model.fit(X_train, y_train)
    return model, {
        "train_mae": mean_absolute_error(y_train, model.predict(X_train)),
        "test_mae": mean_absolute_error(y_test, model.predict(X_test)),
    }


def predict(model: DecisionTreeRegressor, X: pd.DataFrame) -> np.ndarray:
    """Return predictions for X."""
    return model.predict(X)


def run(csv_path: str | None = None) -> dict:
    """End-to-end: load → preprocess → train → return metrics."""
    df = load_data(csv_path)
    X, Y = preprocess(df)
    _, metrics = train(X, Y)
    return metrics
