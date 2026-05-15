![CI](https://github.com/atharvadevne123/Medical-Insurance-Cost-Prediction/actions/workflows/ci.yml/badge.svg)
![Docker](https://github.com/atharvadevne123/Medical-Insurance-Cost-Prediction/actions/workflows/docker-publish.yml/badge.svg)
![Python Package](https://github.com/atharvadevne123/Medical-Insurance-Cost-Prediction/actions/workflows/python-publish.yml/badge.svg)

# Medical Insurance Cost Prediction

Predict annual medical insurance charges from demographic and health features. Exposes a production-ready **FastAPI** service with ensemble ML models (Decision Tree, Random Forest, Ridge Regression), input validation, KS-test drift monitoring, and a full test suite.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/atharvadevne123/Medical-Insurance-Cost-Prediction
cd Medical-Insurance-Cost-Prediction

# 2. Install
make install

# 3. Run API server
make run
# → http://localhost:8000/docs

# 4. Or with Docker
make docker-up
```

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"sex":"male","bmi":27.5,"children":2,"smoker":"no","region":"northeast"}'
# → {"predicted_charge": 6420.18, "model_version": "v1-decision-tree"}
```

---

## Dataset

| Column     | Type        | Description                              |
|------------|-------------|------------------------------------------|
| `age`      | int         | Age of primary beneficiary (0–120)       |
| `sex`      | categorical | `male` / `female`                        |
| `bmi`      | float       | Body Mass Index (0–100)                  |
| `children` | int         | Number of dependents (0–20)              |
| `smoker`   | categorical | `yes` / `no`                             |
| `region`   | categorical | `northeast` / `northwest` / `southeast` / `southwest` |
| `charges`  | float       | **Target** — annual insurance cost (USD) |

Source: `insurance_predictor/data/insurance.csv` (1,338 rows, no missing values).

---

## Architecture

```
Medical-Insurance-Cost-Prediction/
├── app/
│   ├── main.py          # FastAPI application (lifespan, endpoints, middleware)
│   ├── schemas.py       # Pydantic request/response models
│   ├── models.py        # Ensemble: DecisionTree, RandomForest, Ridge
│   ├── monitoring.py    # KS-test drift detection (DriftMonitor)
│   └── persistence.py   # joblib save/load helpers
├── insurance_predictor/
│   ├── predictor.py     # Core: load_data, preprocess, train, predict, run
│   └── data/insurance.csv
├── tests/               # pytest test suite (9 modules, 80+ tests)
├── scripts/
│   └── train_pipeline.py  # CLI: train all models, save best
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

**Request flow**: `POST /predict` → Pydantic validation → `preprocess()` → trained model → response.

---

## API Reference

### `GET /health`
Returns service health and whether the model is loaded.

```json
{"status": "healthy", "model_loaded": true, "version": "1.0.0"}
```

### `GET /metrics`
Returns model performance from the last training run.

```json
{"train_mae": 1823.4, "test_mae": 2541.7, "test_rmse": 4312.5, "test_r2": 0.857, "model_version": "v1-decision-tree"}
```

### `GET /version`
Returns API and Python version info.

### `POST /predict`
Predict insurance charge for one individual.

**Request body:**
```json
{
  "age": 35, "sex": "male", "bmi": 27.5,
  "children": 2, "smoker": "no", "region": "northeast"
}
```

**Response:**
```json
{"predicted_charge": 6420.18, "model_version": "v1-decision-tree"}
```

### `POST /predict/batch`
Predict for up to 1000 individuals.

```json
{"records": [{"age": 25, "sex": "female", "bmi": 22.0, "children": 0, "smoker": "no", "region": "southwest"}]}
```

---

## ML Workflow

1. **Load** — `insurance_predictor.predictor.load_data()` validates schema and raises `FileNotFoundError` / `ValueError` on bad input.
2. **Preprocess** — `StandardScaler` on continuous features, one-hot encoding on categorical features.
3. **Train** — Three models evaluated; best by test MAE is served.
4. **Monitor** — `DriftMonitor` compares incoming feature distributions to training baseline using the two-sample KS test.

---

## Testing

```bash
make test
# Or:
PYTHONPATH=. pytest tests/ -v
```

| Test module             | What it covers                            |
|-------------------------|-------------------------------------------|
| `test_data_loading.py`  | Schema, nulls, value ranges               |
| `test_preprocessing.py` | Encoding, scaling, shape preservation     |
| `test_training.py`      | Metrics, reproducibility, hyperparameters |
| `test_prediction.py`    | Array shape, positivity, batch sizes      |
| `test_api.py`           | All endpoints, validation rejection       |
| `test_models.py`        | Ensemble model comparison                 |
| `test_monitoring.py`    | KS drift detection                        |
| `test_persistence.py`   | Save/load/roundtrip                       |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run `make lint` and `make test` before opening a PR.

---

## Author

Atharva Devne
