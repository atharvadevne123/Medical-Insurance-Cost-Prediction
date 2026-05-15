"""Custom exception classes for the insurance predictor API."""
from __future__ import annotations


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is attempted before the model is initialised.

    This typically indicates the application startup lifespan did not complete
    successfully — check logs for the root cause.
    """


class PredictionError(RuntimeError):
    """Raised when the model raises an unexpected error during inference.

    Attributes:
        original: The original exception that caused the prediction to fail.
    """

    def __init__(self, message: str, original: Exception | None = None) -> None:
        super().__init__(message)
        self.original = original


class DataValidationError(ValueError):
    """Raised when input data fails business-rule validation beyond Pydantic.

    Examples include detected distribution shift, out-of-population values,
    or logical inconsistencies across fields.
    """
