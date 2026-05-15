"""Ensemble model: Decision Tree, Random Forest, and Ridge Regression."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)

ModelType = DecisionTreeRegressor | RandomForestRegressor | Ridge


def train_decision_tree(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    max_depth: int = 6,
    min_samples_split: int = 9,
    test_size: float = 0.2,
    random_state: int = 20,
) -> tuple[DecisionTreeRegressor, dict[str, float]]:
    """Train a Decision Tree regressor.

    Args:
        X: Feature matrix.
        Y: Target charges DataFrame.
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples for a split.
        test_size: Fraction held out for evaluation.
        random_state: Reproducibility seed.

    Returns:
        Tuple (fitted model, metrics dict).
    """
    return _train_model(
        DecisionTreeRegressor(
            criterion="absolute_error",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=21,
        ),
        X, Y, test_size, random_state,
    )


def train_random_forest(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int = 8,
    test_size: float = 0.2,
    random_state: int = 20,
) -> tuple[RandomForestRegressor, dict[str, float]]:
    """Train a Random Forest regressor.

    Args:
        X: Feature matrix.
        Y: Target charges DataFrame.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of each tree.
        test_size: Fraction held out for evaluation.
        random_state: Reproducibility seed.

    Returns:
        Tuple (fitted model, metrics dict).
    """
    return _train_model(
        RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        ),
        X, Y, test_size, random_state,
    )


def train_ridge(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    alpha: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 20,
) -> tuple[Ridge, dict[str, float]]:
    """Train a Ridge regression model.

    Args:
        X: Feature matrix.
        Y: Target charges DataFrame.
        alpha: Regularisation strength.
        test_size: Fraction held out for evaluation.
        random_state: Reproducibility seed.

    Returns:
        Tuple (fitted model, metrics dict).
    """
    return _train_model(
        Ridge(alpha=alpha),
        X, Y, test_size, random_state,
    )


def _train_model(
    model: Any,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[Any, dict[str, float]]:
    """Generic training loop shared by all model types."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    metrics: dict[str, float] = {
        "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "test_r2": float(r2_score(y_test, y_pred)),
    }
    logger.info(
        "%s — test_mae=%.2f test_r2=%.4f",
        model.__class__.__name__,
        metrics["test_mae"],
        metrics["test_r2"],
    )
    return model, metrics


def best_model(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 20,
) -> tuple[Any, dict[str, float], str]:
    """Train all models and return the one with the lowest test MAE.

    Args:
        X: Feature matrix.
        Y: Target charges DataFrame.
        test_size: Fraction held out for evaluation.
        random_state: Reproducibility seed.

    Returns:
        Tuple (best fitted model, its metrics, model name).
    """
    candidates = {
        "decision_tree": train_decision_tree(X, Y, test_size=test_size, random_state=random_state),
        "random_forest": train_random_forest(X, Y, test_size=test_size, random_state=random_state),
        "ridge": train_ridge(X, Y, test_size=test_size, random_state=random_state),
    }
    best_name = min(candidates, key=lambda k: candidates[k][1]["test_mae"])
    best_mdl, best_metrics = candidates[best_name]
    logger.info("Best model: %s with test_mae=%.2f", best_name, best_metrics["test_mae"])
    return best_mdl, best_metrics, best_name
