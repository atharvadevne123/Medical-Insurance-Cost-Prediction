"""Tests for custom exception hierarchy."""
from __future__ import annotations

import pytest

from app.exceptions import DataValidationError, ModelNotLoadedError, PredictionError


class TestModelNotLoadedError:
    def test_is_runtime_error(self) -> None:
        with pytest.raises(RuntimeError):
            raise ModelNotLoadedError("model not loaded")

    def test_message_preserved(self) -> None:
        exc = ModelNotLoadedError("test message")
        assert "test message" in str(exc)


class TestPredictionError:
    def test_is_runtime_error(self) -> None:
        with pytest.raises(RuntimeError):
            raise PredictionError("prediction failed")

    def test_original_is_none_by_default(self) -> None:
        exc = PredictionError("failed")
        assert exc.original is None

    def test_stores_original_exception(self) -> None:
        original = ValueError("bad input")
        exc = PredictionError("failed", original=original)
        assert exc.original is original


class TestDataValidationError:
    def test_is_value_error(self) -> None:
        with pytest.raises(ValueError):
            raise DataValidationError("bad data")

    def test_message_preserved(self) -> None:
        exc = DataValidationError("drift detected")
        assert "drift" in str(exc)
