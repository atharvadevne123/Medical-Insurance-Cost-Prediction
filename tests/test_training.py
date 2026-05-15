"""Tests for model training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from insurance_predictor.predictor import train


class TestTrain:
    def test_returns_tuple(self, preprocessed) -> None:
        X, Y = preprocessed
        result = train(X, Y)
        assert isinstance(result, tuple) and len(result) == 2

    def test_returns_decision_tree(self, preprocessed) -> None:
        X, Y = preprocessed
        model, _ = train(X, Y)
        assert isinstance(model, DecisionTreeRegressor)

    def test_metrics_dict_keys(self, preprocessed) -> None:
        X, Y = preprocessed
        _, metrics = train(X, Y)
        assert "train_mae" in metrics
        assert "test_mae" in metrics

    def test_mae_is_positive(self, preprocessed) -> None:
        X, Y = preprocessed
        _, metrics = train(X, Y)
        assert metrics["train_mae"] >= 0
        assert metrics["test_mae"] >= 0

    def test_train_mae_reasonable(self, preprocessed) -> None:
        X, Y = preprocessed
        _, metrics = train(X, Y)
        # MAE should be less than the mean of charges (basic sanity)
        assert metrics["train_mae"] < Y["charges"].mean()

    def test_model_is_fitted(self, preprocessed) -> None:
        X, Y = preprocessed
        model, _ = train(X, Y)
        # Fitted model has tree_ attribute populated
        assert model.tree_ is not None

    def test_model_feature_count(self, preprocessed) -> None:
        X, Y = preprocessed
        model, _ = train(X, Y)
        assert model.n_features_in_ == X.shape[1]

    @pytest.mark.parametrize("max_depth", [3, 5, 8])
    def test_varying_max_depth(self, preprocessed, max_depth: int) -> None:
        X, Y = preprocessed
        model, metrics = train(X, Y, max_depth=max_depth)
        assert model.max_depth == max_depth
        assert metrics["test_mae"] >= 0

    @pytest.mark.parametrize("test_size", [0.1, 0.2, 0.3])
    def test_varying_test_size(self, preprocessed, test_size: float) -> None:
        X, Y = preprocessed
        _, metrics = train(X, Y, test_size=test_size)
        assert metrics["test_mae"] >= 0

    def test_reproducible_with_same_seed(self, preprocessed) -> None:
        X, Y = preprocessed
        _, m1 = train(X, Y, random_state=42)
        _, m2 = train(X, Y, random_state=42)
        assert abs(m1["test_mae"] - m2["test_mae"]) < 1e-6

    def test_different_seeds_may_differ(self, preprocessed) -> None:
        X, Y = preprocessed
        _, m1 = train(X, Y, random_state=1)
        _, m2 = train(X, Y, random_state=99)
        # Not identical (very unlikely to be same with different seeds)
        assert isinstance(m1["test_mae"], float)
        assert isinstance(m2["test_mae"], float)
