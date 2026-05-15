"""Tests for business-rule validation."""
from __future__ import annotations

import pytest

from app.exceptions import DataValidationError
from app.schemas import PredictRequest
from app.validation import validate_batch_size, validate_predict_request

_VALID = {
    "age": 35, "sex": "male", "bmi": 27.5,
    "children": 2, "smoker": "no", "region": "northeast",
}


class TestValidatePredictRequest:
    def test_valid_request_passes(self) -> None:
        req = PredictRequest(**_VALID)
        validate_predict_request(req)  # should not raise

    def test_too_many_children_raises(self) -> None:
        req = PredictRequest(**{**_VALID, "children": 11})
        with pytest.raises(DataValidationError, match="children"):
            validate_predict_request(req)

    @pytest.mark.parametrize("children", [0, 5, 10])
    def test_valid_children_counts(self, children: int) -> None:
        req = PredictRequest(**{**_VALID, "children": children})
        validate_predict_request(req)  # should not raise


class TestValidateBatchSize:
    def test_valid_size_passes(self) -> None:
        validate_batch_size(5)  # should not raise

    def test_empty_batch_raises(self) -> None:
        with pytest.raises(DataValidationError, match="at least one"):
            validate_batch_size(0)

    def test_oversized_batch_raises(self) -> None:
        with pytest.raises(DataValidationError, match="exceeds"):
            validate_batch_size(1001)

    @pytest.mark.parametrize("n", [1, 100, 1000])
    def test_boundary_values(self, n: int) -> None:
        validate_batch_size(n)  # should not raise
