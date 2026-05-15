"""Pytest fixtures for Medical-Insurance-Cost-Prediction tests."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from insurance_predictor.predictor import load_data, preprocess

DATA_DIR = pathlib.Path(__file__).parent.parent / "insurance_predictor" / "data"


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Load the real insurance dataset once per session."""
    return load_data()


@pytest.fixture(scope="session")
def preprocessed(sample_df: pd.DataFrame):
    """Return preprocessed (X, Y) once per session."""
    return preprocess(sample_df)


@pytest.fixture
def small_df() -> pd.DataFrame:
    """Minimal synthetic DataFrame for fast unit tests."""
    return pd.DataFrame(
        {
            "age": [25, 45, 35, 55, 30],
            "sex": ["male", "female", "male", "female", "male"],
            "bmi": [22.0, 28.5, 31.2, 24.1, 19.8],
            "children": [0, 2, 1, 3, 0],
            "smoker": ["no", "yes", "no", "no", "yes"],
            "region": ["northeast", "southwest", "northwest", "southeast", "northeast"],
            "charges": [3000.0, 15000.0, 6000.0, 8000.0, 12000.0],
        }
    )


@pytest.fixture
def small_preprocessed(small_df: pd.DataFrame):
    """Return preprocessed (X, Y) for the small synthetic dataset."""
    return preprocess(small_df)
