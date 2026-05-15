# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- FastAPI service with `/predict`, `/predict/batch`, `/health`, `/metrics`, `/version` endpoints
- Pydantic schemas with full input validation
- Ensemble models: Decision Tree, Random Forest, Ridge Regression
- KS-test drift monitoring via `DriftMonitor`
- Model persistence with joblib
- Comprehensive test suite (conftest, data loading, preprocessing, training, prediction, API, monitoring, models, persistence)
- GitHub Actions CI workflow (lint + pytest matrix 3.10–3.12)
- Docker and docker-compose support
- Makefile with install/test/lint/run/docker targets
- `.env.example` with documented environment variables
- `.pre-commit-config.yaml` with ruff and trailing-whitespace hooks
- `CONTRIBUTING.md` and `CHANGELOG.md`

### Changed
- Updated `pyproject.toml` with ruff and pytest configuration
- Refactored `predictor.py` with type annotations, docstrings, and structured logging
- Added `test_rmse` and `test_r2` metrics alongside MAE

### Fixed
- `load_data()` now raises `FileNotFoundError` instead of a cryptic pandas error
- `preprocess()` validates that `charges` column exists before processing

## [1.1.0] - 2026-05-15

### Added
- `app/config.py` — `Settings` dataclass with `lru_cache` singleton
- `app/exceptions.py` — Custom exception hierarchy: `ModelNotLoadedError`, `PredictionError`, `DataValidationError`
- `app/validation.py` — Business-rule validation beyond Pydantic (cross-field checks, batch size)
- `app/logging_config.py` — Centralised `configure_logging()` and `get_logger()` helpers
- `app/utils.py` — `round_to_cents`, `clamp`, `bmi_category` (lru_cache), `dataframe_hash`, `flatten_metrics`
- `scripts/evaluate_models.py` — Side-by-side model comparison table
- `scripts/check_data_drift.py` — CLI for KS-test drift comparison between two CSVs
- `scripts/generate_sample_requests.py` — Random prediction request generator
- `tests/test_config.py`, `test_exceptions.py`, `test_validation.py`, `test_utils.py`, `test_logging_config.py`, `test_middleware.py`, `test_feature_importance.py`, `test_integration.py`, `test_scripts.py`

## [1.0.0] - 2024-01-01

### Added
- Initial notebook: Medical Insurance Cost Prediction using Decision Tree Regression
- Bundled `insurance.csv` dataset
- `insurance_predictor` Python package extracted from notebook
