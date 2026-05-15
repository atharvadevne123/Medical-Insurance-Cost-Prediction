"""Drift detection for insurance prediction inputs using the KS test."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

DEFAULT_P_THRESHOLD = 0.05


@dataclass
class DriftReport:
    """Result of a drift detection check."""

    feature: str
    ks_statistic: float
    p_value: float
    drifted: bool
    threshold: float


@dataclass
class DriftMonitor:
    """Monitor feature distributions for covariate shift using the KS test.

    Attributes:
        reference_data: Baseline feature distributions from training data.
        p_threshold: P-value below which drift is declared.
    """

    reference_data: Optional[pd.DataFrame] = None
    p_threshold: float = DEFAULT_P_THRESHOLD
    _reports: list[DriftReport] = field(default_factory=list)

    def fit(self, reference_df: pd.DataFrame) -> "DriftMonitor":
        """Store the reference distribution for future comparisons.

        Args:
            reference_df: Training-time feature DataFrame.

        Returns:
            Self for method chaining.
        """
        numeric_cols = reference_df.select_dtypes(include=[np.number]).columns
        self.reference_data = reference_df[numeric_cols].copy()
        logger.info("DriftMonitor fitted on %d rows, %d features", len(reference_df), len(numeric_cols))
        return self

    def check(self, current_df: pd.DataFrame) -> list[DriftReport]:
        """Run KS tests between reference and current distributions.

        Args:
            current_df: Incoming batch of feature data.

        Returns:
            List of DriftReport objects, one per numeric feature.

        Raises:
            RuntimeError: If the monitor has not been fitted.
        """
        if self.reference_data is None:
            raise RuntimeError("DriftMonitor must be fitted before calling check()")
        reports: list[DriftReport] = []
        for col in self.reference_data.columns:
            if col not in current_df.columns:
                continue
            ref = self.reference_data[col].dropna().values
            cur = current_df[col].dropna().values
            if len(cur) < 2:
                continue
            ks_stat, p_val = stats.ks_2samp(ref, cur)
            drifted = p_val < self.p_threshold
            report = DriftReport(
                feature=col,
                ks_statistic=float(ks_stat),
                p_value=float(p_val),
                drifted=drifted,
                threshold=self.p_threshold,
            )
            reports.append(report)
            if drifted:
                logger.warning(
                    "Drift detected in '%s': ks=%.4f p=%.4f", col, ks_stat, p_val
                )
        self._reports = reports
        return reports

    @property
    def any_drift(self) -> bool:
        """True if any feature in the last check showed drift."""
        return any(r.drifted for r in self._reports)

    def summary(self) -> pd.DataFrame:
        """Return the last check results as a DataFrame."""
        return pd.DataFrame([
            {
                "feature": r.feature,
                "ks_statistic": r.ks_statistic,
                "p_value": r.p_value,
                "drifted": r.drifted,
            }
            for r in self._reports
        ])
