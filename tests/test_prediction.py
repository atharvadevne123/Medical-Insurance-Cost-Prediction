"""Tests for prediction functionality."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_predictor.predictor import predict, run, train


class TestPredict:
    def test_returns_array(self, preprocessed) -> None:
        X, Y = preprocessed
        model, _ = train(X, Y)
        result = predict(model, X)
        assert isinstance(result, np.ndarray)

    def test_shape_matches_input(self, preprocessed) -> None:
        X, Y = preprocessed
        model, _ = train(X, Y)
        result = predict(model, X)
        assert len(result) == len(X)

    def test_predictions_are_positive(self, preprocessed) -> None:
        X, Y = preprocessed
        model, _ = train(X, Y)
        result = predict(model, X)
        # Insurance costs should not be negative
        assert (result > 0).all()

    def test_predictions_are_finite(self, preprocessed) -> None:
        X, Y = preprocessed
        model, _ = train(X, Y)
        result = predict(model, X)
        assert np.isfinite(result).all()

    def test_single_row_prediction(self, preprocessed) -> None:
        X, Y = preprocessed
        model, _ = train(X, Y)
        result = predict(model, X.iloc[:1])
        assert len(result) == 1

    @pytest.mark.parametrize("n_rows", [1, 5, 50])
    def test_batch_sizes(self, preprocessed, n_rows: int) -> None:
        X, Y = preprocessed
        model, _ = train(X, Y)
        result = predict(model, X.iloc[:n_rows])
        assert len(result) == n_rows


class TestRun:
    def test_returns_dict(self) -> None:
        result = run()
        assert isinstance(result, dict)

    def test_has_train_mae(self) -> None:
        result = run()
        assert "train_mae" in result

    def test_has_test_mae(self) -> None:
        result = run()
        assert "test_mae" in result

    def test_mae_values_are_numeric(self) -> None:
        result = run()
        assert isinstance(result["train_mae"], float)
        assert isinstance(result["test_mae"], float)

    def test_run_with_custom_path(self) -> None:
        import pathlib
        csv_path = str(
            pathlib.Path(__file__).parent.parent / "insurance_predictor" / "data" / "insurance.csv"
        )
        result = run(csv_path)
        assert result["test_mae"] > 0

    def test_run_returns_r2(self) -> None:
        result = run()
        assert "test_r2" in result

    def test_run_returns_rmse(self) -> None:
        result = run()
        assert "test_rmse" in result

    def test_run_r2_positive(self) -> None:
        result = run()
        assert result["test_r2"] > 0
