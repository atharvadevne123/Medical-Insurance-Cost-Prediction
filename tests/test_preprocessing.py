"""Tests for the preprocessing pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_predictor.predictor import preprocess


class TestPreprocess:
    def test_returns_tuple(self, small_df: pd.DataFrame) -> None:
        result = preprocess(small_df)
        assert isinstance(result, tuple) and len(result) == 2

    def test_X_is_dataframe(self, small_preprocessed) -> None:
        X, _ = small_preprocessed
        assert isinstance(X, pd.DataFrame)

    def test_Y_is_dataframe(self, small_preprocessed) -> None:
        _, Y = small_preprocessed
        assert isinstance(Y, pd.DataFrame)

    def test_charges_removed_from_X(self, small_preprocessed) -> None:
        X, _ = small_preprocessed
        assert "charges" not in X.columns

    def test_Y_contains_charges(self, small_preprocessed) -> None:
        _, Y = small_preprocessed
        assert "charges" in Y.columns

    def test_no_object_columns_in_X(self, small_preprocessed) -> None:
        X, _ = small_preprocessed
        object_cols = [c for c in X.columns if X[c].dtype == object]
        assert object_cols == []

    def test_continuous_columns_scaled(self, small_preprocessed) -> None:
        X, _ = small_preprocessed
        # After StandardScaler, continuous columns should not be in original range
        assert "age" in X.columns
        assert X["age"].abs().max() < 10

    def test_categorical_columns_one_hot_encoded(self, small_preprocessed) -> None:
        X, _ = small_preprocessed
        assert any("smoker" in c for c in X.columns)

    def test_row_count_preserved(self, small_df: pd.DataFrame) -> None:
        X, Y = preprocess(small_df)
        assert len(X) == len(small_df)
        assert len(Y) == len(small_df)

    def test_full_dataset_shape(self, preprocessed) -> None:
        X, Y = preprocessed
        assert X.shape[0] == Y.shape[0]
        assert X.shape[1] > 0

    def test_no_nan_in_X(self, preprocessed) -> None:
        X, _ = preprocessed
        assert not X.isnull().any().any()

    def test_no_nan_in_Y(self, preprocessed) -> None:
        _, Y = preprocessed
        assert not Y.isnull().any().any()

    @pytest.mark.parametrize("smoker_val,expected_col", [
        ("yes", "smoker_yes"),
        ("no", "smoker_no"),
    ])
    def test_smoker_one_hot_present(
        self, small_df: pd.DataFrame, smoker_val: str, expected_col: str
    ) -> None:
        X, _ = preprocess(small_df)
        assert expected_col in X.columns
