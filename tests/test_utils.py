"""Tests for shared utility functions."""
from __future__ import annotations

import pandas as pd
import pytest

from app.utils import bmi_category, clamp, dataframe_hash, flatten_metrics, round_to_cents


class TestRoundToCents:
    @pytest.mark.parametrize("value,expected", [
        (1234.567, 1234.57),
        (0.001, 0.0),
        (100.0, 100.0),
        (-5.555, -5.56),
    ])
    def test_rounding(self, value: float, expected: float) -> None:
        assert round_to_cents(value) == expected


class TestClamp:
    def test_within_range(self) -> None:
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_below_lo(self) -> None:
        assert clamp(-1.0, 0.0, 10.0) == 0.0

    def test_above_hi(self) -> None:
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_at_lo(self) -> None:
        assert clamp(0.0, 0.0, 10.0) == 0.0

    def test_at_hi(self) -> None:
        assert clamp(10.0, 0.0, 10.0) == 10.0


class TestBmiCategory:
    @pytest.mark.parametrize("bmi,expected", [
        (15.0, "Underweight"),
        (22.0, "Normal"),
        (27.0, "Overweight"),
        (35.0, "Obese"),
        (18.4, "Underweight"),
        (18.5, "Normal"),
        (25.0, "Overweight"),
        (30.0, "Obese"),
    ])
    def test_categories(self, bmi: float, expected: str) -> None:
        assert bmi_category(bmi) == expected


class TestDataframeHash:
    def test_same_df_same_hash(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert dataframe_hash(df) == dataframe_hash(df)

    def test_different_df_different_hash(self) -> None:
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [1, 3]})
        assert dataframe_hash(df1) != dataframe_hash(df2)

    def test_returns_string(self) -> None:
        df = pd.DataFrame({"x": [1]})
        assert isinstance(dataframe_hash(df), str)

    def test_hash_length(self) -> None:
        df = pd.DataFrame({"x": [1]})
        assert len(dataframe_hash(df)) == 64


class TestFlattenMetrics:
    def test_flat_dict_unchanged(self) -> None:
        m = {"mae": 100.0, "r2": 0.85}
        assert flatten_metrics(m) == m

    def test_nested_dict_flattened(self) -> None:
        m = {"train": {"mae": 100.0}, "test": {"mae": 200.0}}
        flat = flatten_metrics(m)
        assert "train.mae" in flat
        assert "test.mae" in flat

    def test_values_are_float(self) -> None:
        m = {"a": 1, "b": 2}
        flat = flatten_metrics(m)
        assert all(isinstance(v, float) for v in flat.values())
