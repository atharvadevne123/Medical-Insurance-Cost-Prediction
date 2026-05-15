"""Tests for FastAPI endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a test client with the FastAPI app."""
    from app.main import app
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_field(self, client: TestClient) -> None:
        r = client.get("/health")
        assert "status" in r.json()

    def test_health_model_loaded(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_status_healthy(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "healthy"


class TestVersion:
    def test_version_returns_200(self, client: TestClient) -> None:
        assert client.get("/version").status_code == 200

    def test_version_has_fields(self, client: TestClient) -> None:
        data = client.get("/version").json()
        assert "version" in data
        assert "model_version" in data
        assert "python_version" in data


class TestMetrics:
    def test_metrics_returns_200(self, client: TestClient) -> None:
        assert client.get("/metrics").status_code == 200

    def test_metrics_has_mae(self, client: TestClient) -> None:
        data = client.get("/metrics").json()
        assert "test_mae" in data
        assert "train_mae" in data

    def test_metrics_has_r2(self, client: TestClient) -> None:
        data = client.get("/metrics").json()
        assert "test_r2" in data

    def test_metrics_mae_positive(self, client: TestClient) -> None:
        data = client.get("/metrics").json()
        assert data["test_mae"] > 0


class TestPredict:
    _valid_payload = {
        "age": 35,
        "sex": "male",
        "bmi": 27.5,
        "children": 2,
        "smoker": "no",
        "region": "northeast",
    }

    def test_predict_returns_200(self, client: TestClient) -> None:
        r = client.post("/predict", json=self._valid_payload)
        assert r.status_code == 200

    def test_predict_has_charge(self, client: TestClient) -> None:
        data = client.post("/predict", json=self._valid_payload).json()
        assert "predicted_charge" in data

    def test_predict_charge_positive(self, client: TestClient) -> None:
        data = client.post("/predict", json=self._valid_payload).json()
        assert data["predicted_charge"] > 0

    def test_predict_smoker_higher_charge(self, client: TestClient) -> None:
        non_smoker = dict(self._valid_payload, smoker="no")
        smoker = dict(self._valid_payload, smoker="yes")
        charge_non = client.post("/predict", json=non_smoker).json()["predicted_charge"]
        charge_smoker = client.post("/predict", json=smoker).json()["predicted_charge"]
        assert charge_smoker > charge_non

    @pytest.mark.parametrize("field,bad_value", [
        ("age", -1),
        ("age", 200),
        ("bmi", -5),
        ("sex", "unknown"),
        ("smoker", "maybe"),
        ("region", "mars"),
    ])
    def test_predict_invalid_input(
        self, client: TestClient, field: str, bad_value
    ) -> None:
        payload = dict(self._valid_payload, **{field: bad_value})
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_predict_returns_correlation_id(self, client: TestClient) -> None:
        r = client.post("/predict", json=self._valid_payload)
        assert "x-correlation-id" in r.headers


class TestBatchPredict:
    _records = [
        {"age": 25, "sex": "female", "bmi": 22.0, "children": 0, "smoker": "no", "region": "southwest"},
        {"age": 50, "sex": "male", "bmi": 35.0, "children": 3, "smoker": "yes", "region": "southeast"},
    ]

    def test_batch_predict_returns_200(self, client: TestClient) -> None:
        r = client.post("/predict/batch", json={"records": self._records})
        assert r.status_code == 200

    def test_batch_predict_count_matches(self, client: TestClient) -> None:
        data = client.post("/predict/batch", json={"records": self._records}).json()
        assert data["count"] == len(self._records)
        assert len(data["predictions"]) == len(self._records)

    def test_batch_predict_empty_records_fails(self, client: TestClient) -> None:
        r = client.post("/predict/batch", json={"records": []})
        assert r.status_code == 422

    @pytest.mark.parametrize("n", [1, 5, 10])
    def test_batch_predict_various_sizes(self, client: TestClient, n: int) -> None:
        records = [
            {"age": 30 + i, "sex": "male", "bmi": 25.0, "children": 0, "smoker": "no", "region": "northeast"}
            for i in range(n)
        ]
        data = client.post("/predict/batch", json={"records": records}).json()
        assert data["count"] == n


class TestCorrelationId:
    def test_response_has_correlation_id(self, client: TestClient) -> None:
        r = client.get("/health")
        assert "x-correlation-id" in r.headers

    def test_custom_correlation_id_echoed(self, client: TestClient) -> None:
        cid = "test-correlation-123"
        r = client.get("/health", headers={"X-Correlation-ID": cid})
        assert r.headers.get("x-correlation-id") == cid

    def test_response_time_header_present(self, client: TestClient) -> None:
        r = client.get("/health")
        assert "x-response-time" in r.headers
