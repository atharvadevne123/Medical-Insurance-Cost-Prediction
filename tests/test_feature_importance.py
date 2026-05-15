"""Tests for feature importance analysis."""
from __future__ import annotations

import pandas as pd
import pytest

from insurance_predictor.predictor import feature_importance, train


class TestFeatureImportance:
    def test_returns_series(self, preprocessed) -> None:
        X, Y = preprocessed
        model, _ = train(X, Y)
        result = feature_importance(model, list(X.columns))
        assert isinstance(result, pd.Series)

    def test_length_matches_features(self, preprocessed) -> None:
        X, Y = preprocessed
        model, _ = train(X, Y)
        result = feature_importance(model, list(X.columns))
        assert len(result) == X.shape[1]

    def test_importances_sum_to_one(self, preprocessed) -> None:
        X, Y = preprocessed
        model, _ = train(X, Y)
        result = feature_importance(model, list(X.columns))
        assert abs(result.sum() - 1.0) < 1e-6

    def test_sorted_descending(self, preprocessed) -> None:
        X, Y = preprocessed
        model, _ = train(X, Y)
        result = feature_importance(model, list(X.columns))
        assert (result.values[:-1] >= result.values[1:]).all()

    def test_non_negative(self, preprocessed) -> None:
        X, Y = preprocessed
        model, _ = train(X, Y)
        result = feature_importance(model, list(X.columns))
        assert (result >= 0).all()

    def test_smoker_in_top_features(self, preprocessed) -> None:
        X, Y = preprocessed
        model, _ = train(X, Y)
        result = feature_importance(model, list(X.columns))
        top3 = set(result.index[:3])
        assert any("smoker" in f for f in top3)

    @pytest.mark.parametrize("top_n", [1, 3, 5])
    def test_top_n_subset(self, preprocessed, top_n: int) -> None:
        X, Y = preprocessed
        model, _ = train(X, Y)
        result = feature_importance(model, list(X.columns))
        assert len(result.head(top_n)) == top_n
