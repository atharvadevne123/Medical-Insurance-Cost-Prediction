"""Tests for drift monitoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.monitoring import DriftMonitor, DriftReport


@pytest.fixture
def reference_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "age": rng.normal(40, 10, 500),
        "bmi": rng.normal(28, 5, 500),
        "children": rng.integers(0, 5, 500).astype(float),
    })


@pytest.fixture
def similar_df(reference_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "age": rng.normal(40, 10, 200),
        "bmi": rng.normal(28, 5, 200),
        "children": rng.integers(0, 5, 200).astype(float),
    })


@pytest.fixture
def drifted_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "age": rng.normal(65, 5, 200),
        "bmi": rng.normal(40, 8, 200),
        "children": rng.integers(0, 5, 200).astype(float),
    })


class TestDriftMonitor:
    def test_fit_returns_self(self, reference_df: pd.DataFrame) -> None:
        monitor = DriftMonitor()
        result = monitor.fit(reference_df)
        assert result is monitor

    def test_fit_stores_reference(self, reference_df: pd.DataFrame) -> None:
        monitor = DriftMonitor().fit(reference_df)
        assert monitor.reference_data is not None

    def test_check_returns_list(self, reference_df: pd.DataFrame, similar_df: pd.DataFrame) -> None:
        monitor = DriftMonitor().fit(reference_df)
        reports = monitor.check(similar_df)
        assert isinstance(reports, list)

    def test_check_returns_drift_reports(self, reference_df: pd.DataFrame, similar_df: pd.DataFrame) -> None:
        monitor = DriftMonitor().fit(reference_df)
        reports = monitor.check(similar_df)
        assert all(isinstance(r, DriftReport) for r in reports)

    def test_no_drift_on_similar_data(self, reference_df: pd.DataFrame, similar_df: pd.DataFrame) -> None:
        monitor = DriftMonitor(p_threshold=0.01).fit(reference_df)
        reports = monitor.check(similar_df)
        assert not any(r.drifted for r in reports)

    def test_drift_detected_on_shifted_data(self, reference_df: pd.DataFrame, drifted_df: pd.DataFrame) -> None:
        monitor = DriftMonitor(p_threshold=0.05).fit(reference_df)
        reports = monitor.check(drifted_df)
        assert any(r.drifted for r in reports)

    def test_unfitted_raises(self) -> None:
        monitor = DriftMonitor()
        with pytest.raises(RuntimeError):
            monitor.check(pd.DataFrame({"age": [30]}))

    def test_any_drift_property(self, reference_df: pd.DataFrame, drifted_df: pd.DataFrame) -> None:
        monitor = DriftMonitor().fit(reference_df)
        monitor.check(drifted_df)
        assert isinstance(monitor.any_drift, bool)

    def test_summary_returns_dataframe(self, reference_df: pd.DataFrame, similar_df: pd.DataFrame) -> None:
        monitor = DriftMonitor().fit(reference_df)
        monitor.check(similar_df)
        summary = monitor.summary()
        assert isinstance(summary, pd.DataFrame)
        assert "feature" in summary.columns

    @pytest.mark.parametrize("p_threshold", [0.01, 0.05, 0.10])
    def test_varying_thresholds(
        self, reference_df: pd.DataFrame, drifted_df: pd.DataFrame, p_threshold: float
    ) -> None:
        monitor = DriftMonitor(p_threshold=p_threshold).fit(reference_df)
        reports = monitor.check(drifted_df)
        assert len(reports) > 0
