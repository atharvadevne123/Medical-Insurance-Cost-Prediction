"""Medical insurance cost predictor — extracted from the original notebook."""
from __future__ import annotations

import logging
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)

_DATA = pathlib.Path(__file__).parent / "data" / "insurance.csv"

EXPECTED_COLUMNS = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
CATEGORICAL_COLUMNS = ["sex", "smoker", "region"]
CONTINUOUS_COLUMNS = ["age", "bmi", "children"]


def load_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load the insurance dataset from disk.

    Args:
        csv_path: Path to a CSV file. If None, uses the bundled insurance.csv.

    Returns:
        DataFrame with insurance data.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing.
    """
    path = pathlib.Path(csv_path) if csv_path else _DATA
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    try:
        df = pd.read_csv(str(path))
    except Exception as exc:
        logger.error("Failed to read CSV %s: %s", path, exc)
        raise
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode categorical features and scale continuous ones.

    Args:
        df: Raw insurance DataFrame with a 'charges' column.

    Returns:
        Tuple (X, Y) where X contains engineered features and Y contains charges.

    Raises:
        ValueError: If 'charges' column is absent.
    """
    if "charges" not in df.columns:
        raise ValueError("DataFrame must contain a 'charges' column")
    Y: pd.DataFrame = df[["charges"]].reset_index(drop=True)
    X_raw = df.drop(columns=["charges"]).reset_index(drop=True)
    cat = [c for c in X_raw.columns if X_raw[c].dtype == object]
    con = [c for c in X_raw.columns if X_raw[c].dtype != object]
    ss = StandardScaler()
    X_con = pd.DataFrame(ss.fit_transform(X_raw[con]), columns=con)
    X_cat = pd.get_dummies(X_raw[cat])
    X = X_con.join(X_cat)
    logger.debug("Preprocessed: %d rows, %d features", len(X), X.shape[1])
    return X, Y


def train(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    max_depth: int = 6,
    min_samples_split: int = 9,
    test_size: float = 0.2,
    random_state: int = 20,
) -> tuple[DecisionTreeRegressor, dict[str, float]]:
    """Train a DecisionTreeRegressor and return the fitted model with metrics.

    Args:
        X: Feature matrix.
        Y: Target DataFrame with 'charges' column.
        max_depth: Maximum depth of the decision tree.
        min_samples_split: Minimum samples required to split an internal node.
        test_size: Fraction of data held out for evaluation.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple (model, metrics) where metrics contains train_mae, test_mae, test_rmse,
        and test_r2.
    """
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
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    metrics: dict[str, float] = {
        "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "test_r2": float(r2_score(y_test, y_pred_test)),
    }
    logger.info(
        "Trained model: train_mae=%.2f test_mae=%.2f test_r2=%.4f",
        metrics["train_mae"],
        metrics["test_mae"],
        metrics["test_r2"],
    )
    return model, metrics


def predict(model: DecisionTreeRegressor, X: pd.DataFrame) -> np.ndarray:
    """Run inference on a feature matrix.

    Args:
        model: A fitted DecisionTreeRegressor.
        X: Feature matrix with the same columns used during training.

    Returns:
        1-D array of predicted insurance charges.
    """
    predictions: np.ndarray = model.predict(X)
    return predictions


def feature_importance(model: DecisionTreeRegressor, feature_names: list[str]) -> pd.Series:
    """Return feature importances sorted descending.

    Args:
        model: A fitted DecisionTreeRegressor.
        feature_names: Column names matching model.feature_names_in_.

    Returns:
        pd.Series indexed by feature name, sorted by importance descending.
    """
    return (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
    )


def run(csv_path: Optional[str] = None) -> dict[str, float]:
    """End-to-end pipeline: load → preprocess → train → return metrics.

    Args:
        csv_path: Optional path to a custom CSV file.

    Returns:
        Metrics dict with train_mae, test_mae, test_rmse, test_r2.
    """
    df = load_data(csv_path)
    X, Y = preprocess(df)
    _, metrics = train(X, Y)
    return metrics
