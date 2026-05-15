"""Pydantic schemas for request/response validation."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Input features for a single insurance cost prediction."""

    age: int = Field(..., ge=0, le=120, description="Age of the primary beneficiary")
    sex: Literal["male", "female"] = Field(..., description="Gender")
    bmi: float = Field(..., ge=0.0, le=100.0, description="Body Mass Index")
    children: int = Field(..., ge=0, le=20, description="Number of dependents")
    smoker: Literal["yes", "no"] = Field(..., description="Smoking status")
    region: Literal["northeast", "northwest", "southeast", "southwest"] = Field(
        ..., description="Residential region in the U.S."
    )

    model_config = {"json_schema_extra": {
        "example": {
            "age": 35,
            "sex": "male",
            "bmi": 27.5,
            "children": 2,
            "smoker": "no",
            "region": "northeast",
        }
    }}


class PredictResponse(BaseModel):
    """Predicted insurance charge."""

    predicted_charge: float = Field(..., description="Predicted annual insurance charge in USD")
    model_version: str = Field(..., description="Model version used for prediction")


class BatchPredictRequest(BaseModel):
    """Batch prediction input."""

    records: list[PredictRequest] = Field(..., min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    """Batch prediction output."""

    predictions: list[float]
    count: int
    model_version: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    """Model performance metrics."""

    train_mae: float
    test_mae: float
    test_rmse: float
    test_r2: float
    model_version: str


class VersionResponse(BaseModel):
    """Version information."""

    version: str
    model_version: str
    python_version: str
