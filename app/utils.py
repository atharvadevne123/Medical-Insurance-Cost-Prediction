"""Shared utility functions for the insurance predictor application."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd


def round_to_cents(value: float) -> float:
    """Round a dollar amount to the nearest cent.

    Args:
        value: A monetary value.

    Returns:
        Value rounded to 2 decimal places.
    """
    return round(float(value), 2)


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value to the range [lo, hi].

    Args:
        value: Input value.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Clamped value.
    """
    return max(lo, min(hi, value))


@lru_cache(maxsize=128)
def bmi_category(bmi: float) -> str:
    """Return WHO BMI category string for a given BMI.

    Args:
        bmi: Body Mass Index value.

    Returns:
        One of 'Underweight', 'Normal', 'Overweight', 'Obese'.
    """
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25.0:
        return "Normal"
    if bmi < 30.0:
        return "Overweight"
    return "Obese"


def dataframe_hash(df: pd.DataFrame) -> str:
    """Return a deterministic SHA-256 hash of a DataFrame's contents.

    Useful for cache keys and change detection.

    Args:
        df: Any pandas DataFrame.

    Returns:
        Hex digest string (64 characters).
    """
    serialised = df.to_json(orient="records", date_format="iso")
    return hashlib.sha256(serialised.encode()).hexdigest()


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Recursively flatten a nested metrics dict to a single level.

    Args:
        metrics: Possibly nested dict of metric values.
        prefix: String prepended to every key (used in recursion).

    Returns:
        Flat dict mapping dotted-key paths to float values.
    """
    result: dict[str, float] = {}
    for k, v in metrics.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(flatten_metrics(v, prefix=full_key))
        else:
            result[full_key] = float(v)
    return result
