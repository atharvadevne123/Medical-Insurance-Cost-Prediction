"""Integration tests: end-to-end pipeline and API interactions."""
from __future__ import annotations

import pathlib
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.models import best_model
from app.monitoring import DriftMonitor
from app.persistence import load_model, save_model
from insurance_predictor.predictor import load_data, predict, preprocess, run


class TestEndToEndPipeline:
    def test_run_returns_metrics(self) -> None:
        metrics = run()
        assert isinstance(metrics, dict)
        assert metrics["test_mae"] > 0

    def test_full_pipeline_with_custom_path(self) -> None:
        csv_path = str(
            pathlib.Path(__file__).parent.parent / "insurance_predictor" / "data" / "insurance.csv"
        )
        metrics = run(csv_path)
        assert metrics["test_r2"] > 0.5

    def test_train_and_persist_roundtrip(self) -> None:
        df = load_data()
        X, Y = preprocess(df)
        model, metrics = best_model(X, Y)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "best_model.joblib"
            save_model(model, path)
            loaded = load_model(path)
            preds_original = model.predict(X)
            preds_loaded = loaded.predict(X)
            import numpy as np
            assert np.allclose(preds_original, preds_loaded)


class TestDriftMonitorIntegration:
    def test_fit_and_check_real_data(self) -> None:
        df = load_data()
        X, _ = preprocess(df)
        half = len(X) // 2
        monitor = DriftMonitor().fit(X.iloc[:half])
        reports = monitor.check(X.iloc[half:])
        assert len(reports) > 0

    def test_no_drift_on_same_split(self) -> None:
        df = load_data()
        X, _ = preprocess(df)
        half = len(X) // 2
        monitor = DriftMonitor(p_threshold=0.001).fit(X.iloc[:half])
        reports = monitor.check(X.iloc[:half])
        assert not any(r.drifted for r in reports)


class TestAPIIntegration:
    @pytest.fixture(scope="class")
    def client(self):
        from app.main import app
        with TestClient(app) as c:
            yield c

    def test_health_then_predict_flow(self, client: TestClient) -> None:
        health = client.get("/health").json()
        assert health["model_loaded"] is True
        payload = {
            "age": 40, "sex": "female", "bmi": 25.0,
            "children": 1, "smoker": "no", "region": "southeast",
        }
        result = client.post("/predict", json=payload).json()
        assert result["predicted_charge"] > 0

    def test_metrics_consistency_with_health(self, client: TestClient) -> None:
        metrics = client.get("/metrics").json()
        health = client.get("/health").json()
        assert health["model_loaded"] is True
        assert metrics["test_r2"] > 0

    def test_batch_size_one_matches_single(self, client: TestClient) -> None:
        payload = {
            "age": 30, "sex": "male", "bmi": 24.0,
            "children": 0, "smoker": "no", "region": "northwest",
        }
        single = client.post("/predict", json=payload).json()["predicted_charge"]
        batch = client.post("/predict/batch", json={"records": [payload]}).json()
        assert abs(batch["predictions"][0] - single) < 0.01
