"""Tests for data loading functionality."""
from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from insurance_predictor.predictor import load_data

EXPECTED_COLUMNS = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
CSV_PATH = str(
    pathlib.Path(__file__).parent.parent / "insurance_predictor" / "data" / "insurance.csv"
)


class TestLoadData:
    def test_returns_dataframe(self, sample_df: pd.DataFrame) -> None:
        assert isinstance(sample_df, pd.DataFrame)

    def test_has_expected_columns(self, sample_df: pd.DataFrame) -> None:
        assert EXPECTED_COLUMNS.issubset(set(sample_df.columns))

    def test_not_empty(self, sample_df: pd.DataFrame) -> None:
        assert len(sample_df) > 0

    def test_no_null_values(self, sample_df: pd.DataFrame) -> None:
        assert sample_df.isnull().sum().sum() == 0

    def test_charges_are_positive(self, sample_df: pd.DataFrame) -> None:
        assert (sample_df["charges"] > 0).all()

    def test_age_range(self, sample_df: pd.DataFrame) -> None:
        assert sample_df["age"].between(0, 120).all()

    def test_bmi_range(self, sample_df: pd.DataFrame) -> None:
        assert sample_df["bmi"].between(0, 100).all()

    def test_children_non_negative(self, sample_df: pd.DataFrame) -> None:
        assert (sample_df["children"] >= 0).all()

    def test_smoker_categories(self, sample_df: pd.DataFrame) -> None:
        assert set(sample_df["smoker"].unique()).issubset({"yes", "no"})

    def test_sex_categories(self, sample_df: pd.DataFrame) -> None:
        assert set(sample_df["sex"].unique()).issubset({"male", "female"})

    def test_region_categories(self, sample_df: pd.DataFrame) -> None:
        assert set(sample_df["region"].unique()).issubset(
            {"northeast", "northwest", "southeast", "southwest"}
        )

    def test_custom_csv_path(self) -> None:
        df = load_data(CSV_PATH)
        assert len(df) > 0

    def test_invalid_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_data("/nonexistent/path/data.csv")

    @pytest.mark.parametrize("col", ["age", "bmi", "charges"])
    def test_numeric_columns(self, sample_df: pd.DataFrame, col: str) -> None:
        assert pd.api.types.is_numeric_dtype(sample_df[col])

    def test_row_count_matches_known(self, sample_df: pd.DataFrame) -> None:
        assert len(sample_df) == 1338

    def test_charges_mean_reasonable(self, sample_df: pd.DataFrame) -> None:
        mean_charge = sample_df["charges"].mean()
        assert 5000 < mean_charge < 20000

    def test_age_distribution_has_both_halves(self, sample_df: pd.DataFrame) -> None:
        assert (sample_df["age"] < 40).any()
        assert (sample_df["age"] >= 40).any()

    @pytest.mark.parametrize("region", ["northeast", "northwest", "southeast", "southwest"])
    def test_all_regions_present(self, sample_df: pd.DataFrame, region: str) -> None:
        assert region in sample_df["region"].values
