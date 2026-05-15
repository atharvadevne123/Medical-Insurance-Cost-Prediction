"""Business-rule validation beyond Pydantic schema checks."""
from __future__ import annotations

import logging
from typing import Optional

from app.exceptions import DataValidationError
from app.schemas import PredictRequest

logger = logging.getLogger(__name__)

BMI_OBESITY_THRESHOLD = 40.0
AGE_ELDERLY_THRESHOLD = 75
MAX_CHILDREN = 10


def validate_predict_request(request: PredictRequest) -> None:
    """Apply business-rule validation to a prediction request.

    Pydantic already enforces field types and basic range constraints.
    This function adds cross-field and population-based checks.

    Args:
        request: A Pydantic-validated PredictRequest.

    Raises:
        DataValidationError: If any business rule is violated.
    """
    warnings: list[str] = []

    if request.children > MAX_CHILDREN:
        raise DataValidationError(
            f"children={request.children} exceeds maximum supported value {MAX_CHILDREN}"
        )

    if request.bmi > BMI_OBESITY_THRESHOLD and request.age < 18:
        warnings.append(
            f"Unusually high BMI {request.bmi} for age {request.age} — verify input"
        )

    if request.age > AGE_ELDERLY_THRESHOLD and request.children > 5:
        warnings.append(
            f"age={request.age} with children={request.children} is unusual — verify input"
        )

    for w in warnings:
        logger.warning("Validation warning: %s", w)


def validate_batch_size(n: int, max_size: int = 1000) -> None:
    """Ensure a batch does not exceed the maximum allowed size.

    Args:
        n: Number of records in the batch.
        max_size: Maximum allowed batch size.

    Raises:
        DataValidationError: If n exceeds max_size.
    """
    if n > max_size:
        raise DataValidationError(
            f"Batch size {n} exceeds maximum allowed size {max_size}"
        )
    if n == 0:
        raise DataValidationError("Batch must contain at least one record")
