"""Tests for the ensemble models module."""
from __future__ import annotations

import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from app.models import (
    best_model,
    train_decision_tree,
    train_random_forest,
    train_ridge,
)


class TestTrainDecisionTree:
    def test_returns_decision_tree(self, preprocessed) -> None:
        X, Y = preprocessed
        model, _ = train_decision_tree(X, Y)
        assert isinstance(model, DecisionTreeRegressor)

    def test_metrics_keys(self, preprocessed) -> None:
        X, Y = preprocessed
        _, m = train_decision_tree(X, Y)
        assert {"train_mae", "test_mae", "test_rmse", "test_r2"}.issubset(m)

    def test_r2_reasonable(self, preprocessed) -> None:
        X, Y = preprocessed
        _, m = train_decision_tree(X, Y)
        assert m["test_r2"] > 0.5


class TestTrainRandomForest:
    def test_returns_random_forest(self, preprocessed) -> None:
        X, Y = preprocessed
        model, _ = train_random_forest(X, Y, n_estimators=10)
        assert isinstance(model, RandomForestRegressor)

    def test_metrics_positive(self, preprocessed) -> None:
        X, Y = preprocessed
        _, m = train_random_forest(X, Y, n_estimators=10)
        assert m["test_mae"] > 0
        assert m["test_rmse"] > 0

    def test_r2_better_than_tree(self, preprocessed) -> None:
        X, Y = preprocessed
        _, tree_m = train_decision_tree(X, Y)
        _, rf_m = train_random_forest(X, Y, n_estimators=50)
        # Random Forest should generally outperform single tree
        assert rf_m["test_r2"] > 0.7

    @pytest.mark.parametrize("n_estimators", [5, 10, 20])
    def test_n_estimators_param(self, preprocessed, n_estimators: int) -> None:
        X, Y = preprocessed
        model, _ = train_random_forest(X, Y, n_estimators=n_estimators)
        assert model.n_estimators == n_estimators


class TestTrainRidge:
    def test_returns_ridge(self, preprocessed) -> None:
        X, Y = preprocessed
        model, _ = train_ridge(X, Y)
        assert isinstance(model, Ridge)

    def test_metrics_keys(self, preprocessed) -> None:
        X, Y = preprocessed
        _, m = train_ridge(X, Y)
        assert "test_mae" in m

    @pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
    def test_varying_alpha(self, preprocessed, alpha: float) -> None:
        X, Y = preprocessed
        model, m = train_ridge(X, Y, alpha=alpha)
        assert model.alpha == alpha
        assert m["test_mae"] > 0


class TestBestModel:
    def test_returns_triple(self, preprocessed) -> None:
        X, Y = preprocessed
        result = best_model(X, Y)
        assert len(result) == 3

    def test_model_name_is_string(self, preprocessed) -> None:
        X, Y = preprocessed
        _, _, name = best_model(X, Y)
        assert isinstance(name, str)

    def test_model_name_valid(self, preprocessed) -> None:
        X, Y = preprocessed
        _, _, name = best_model(X, Y)
        assert name in {"decision_tree", "random_forest", "ridge"}

    def test_best_metrics_reasonable(self, preprocessed) -> None:
        X, Y = preprocessed
        _, metrics, _ = best_model(X, Y)
        assert metrics["test_r2"] > 0.5
