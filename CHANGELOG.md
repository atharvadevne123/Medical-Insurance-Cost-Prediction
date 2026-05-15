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

## [1.0.0] - 2024-01-01

### Added
- Initial notebook: Medical Insurance Cost Prediction using Decision Tree Regression
- Bundled `insurance.csv` dataset
- `insurance_predictor` Python package extracted from notebook
