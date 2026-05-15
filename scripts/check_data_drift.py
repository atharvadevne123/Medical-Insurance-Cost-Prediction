"""CLI script to check for data drift between a reference and new dataset."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import logging

import pandas as pd

from app.monitoring import DriftMonitor
from insurance_predictor.predictor import load_data, preprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("check_data_drift")


def main(reference_path: str | None, current_path: str, p_threshold: float) -> int:
    """Compare feature distributions between two datasets.

    Args:
        reference_path: Path to reference CSV (defaults to bundled insurance.csv).
        current_path: Path to new incoming data CSV.
        p_threshold: P-value threshold for declaring drift.

    Returns:
        Exit code: 0 = no drift, 1 = drift detected, 2 = error.
    """
    try:
        ref_df = load_data(reference_path)
        X_ref, _ = preprocess(ref_df)

        cur_df = pd.read_csv(current_path)
        if "charges" in cur_df.columns:
            X_cur, _ = preprocess(cur_df)
        else:
            cur_df["charges"] = 0.0
            X_cur, _ = preprocess(cur_df)

        monitor = DriftMonitor(p_threshold=p_threshold).fit(X_ref)
        reports = monitor.check(X_cur)
        summary = monitor.summary()
        print(summary.to_string(index=False))

        if monitor.any_drift:
            drifted_features = [r.feature for r in reports if r.drifted]
            logger.warning("Drift detected in: %s", drifted_features)
            return 1
        logger.info("No significant drift detected.")
        return 0
    except Exception as exc:
        logger.error("Failed: %s", exc)
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check feature distribution drift")
    parser.add_argument("current_csv", help="Path to incoming data CSV")
    parser.add_argument("--reference", default=None, help="Reference CSV (default: bundled)")
    parser.add_argument("--threshold", type=float, default=0.05, help="KS p-value threshold")
    args = parser.parse_args()
    sys.exit(main(args.reference, args.current_csv, args.threshold))
