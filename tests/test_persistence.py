"""Tests for model persistence."""
from __future__ import annotations

import pathlib
import tempfile

import pytest
from sklearn.tree import DecisionTreeRegressor

from app.persistence import load_model, model_exists, save_model
from insurance_predictor.predictor import train


@pytest.fixture
def trained_model(preprocessed):
    X, Y = preprocessed
    model, _ = train(X, Y)
    return model


class TestSaveModel:
    def test_saves_file(self, trained_model: DecisionTreeRegressor) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "model.joblib"
            result = save_model(trained_model, path)
            assert result.exists()

    def test_returns_path(self, trained_model: DecisionTreeRegressor) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "model.joblib"
            result = save_model(trained_model, path)
            assert isinstance(result, pathlib.Path)

    def test_creates_parent_dirs(self, trained_model: DecisionTreeRegressor) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "subdir" / "nested" / "model.joblib"
            save_model(trained_model, path)
            assert path.exists()


class TestLoadModel:
    def test_roundtrip(self, trained_model: DecisionTreeRegressor, preprocessed) -> None:
        X, Y = preprocessed
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "model.joblib"
            save_model(trained_model, path)
            loaded = load_model(path)
            original_preds = trained_model.predict(X)
            loaded_preds = loaded.predict(X)
            import numpy as np
            assert np.allclose(original_preds, loaded_preds)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path/model.joblib")


class TestModelExists:
    def test_returns_true_for_existing_file(self, trained_model: DecisionTreeRegressor) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "model.joblib"
            save_model(trained_model, path)
            assert model_exists(path) is True

    def test_returns_false_for_missing_file(self) -> None:
        assert model_exists("/nonexistent/path.joblib") is False
