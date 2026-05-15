"""Training pipeline script: train all models and save the best one."""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from app.models import best_model
from app.persistence import save_model
from insurance_predictor.predictor import load_data, preprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("train_pipeline")


def main(csv_path: str | None = None, output_path: str = "models/model.joblib") -> int:
    """Run the full training pipeline.

    Args:
        csv_path: Optional path to a custom CSV dataset.
        output_path: Where to save the best model.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    try:
        logger.info("Loading data from %s", csv_path or "bundled insurance.csv")
        df = load_data(csv_path)

        logger.info("Preprocessing %d rows...", len(df))
        X, Y = preprocess(df)

        logger.info("Training all models and selecting the best...")
        model, metrics, model_name = best_model(X, Y)

        logger.info(
            "Best model: %s | test_mae=%.2f test_r2=%.4f",
            model_name, metrics["test_mae"], metrics["test_r2"],
        )

        dest = save_model(model, output_path)
        logger.info("Model saved to %s", dest)
        return 0
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train insurance cost prediction model")
    parser.add_argument("--csv", default=None, help="Path to dataset CSV")
    parser.add_argument("--output", default="models/model.joblib", help="Output model path")
    args = parser.parse_args()
    sys.exit(main(args.csv, args.output))
