"""FastAPI application for Medical Insurance Cost Prediction."""
from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
    VersionResponse,
)
from insurance_predictor.predictor import (
    load_data,
    predict,
    preprocess,
    run,
    train,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

APP_VERSION = "1.0.0"
MODEL_VERSION = "v1-decision-tree"

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and clean up on shutdown."""
    logger.info("Loading model on startup...")
    try:
        df = load_data()
        X, Y = preprocess(df)
        model, metrics = train(X, Y)
        _state["model"] = model
        _state["metrics"] = metrics
        _state["feature_names"] = list(X.columns)
        logger.info("Model loaded: test_mae=%.2f test_r2=%.4f", metrics["test_mae"], metrics["test_r2"])
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        _state["model"] = None
        _state["metrics"] = {}
    yield
    _state.clear()
    logger.info("Application shutdown complete.")


app = FastAPI(
    title="Medical Insurance Cost Prediction API",
    description="Predict annual medical insurance charges from demographic and health features.",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    """Attach a unique correlation ID to every request for tracing."""
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time"] = f"{elapsed:.4f}s"
    logger.info(
        "%s %s %s %.4fs corr=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
        correlation_id,
    )
    return response


def _build_feature_df(record: PredictRequest) -> pd.DataFrame:
    """Convert a PredictRequest into the feature DataFrame expected by the model."""
    df = pd.DataFrame([{
        "age": record.age,
        "sex": record.sex,
        "bmi": record.bmi,
        "children": record.children,
        "smoker": record.smoker,
        "region": record.region,
        "charges": 0.0,
    }])
    X, _ = preprocess(df)
    model_features: list[str] = _state.get("feature_names", [])
    if model_features:
        for col in model_features:
            if col not in X.columns:
                X[col] = 0
        X = X[model_features]
    return X


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health() -> HealthResponse:
    """Return API health status including whether the model is loaded."""
    model_loaded = _state.get("model") is not None
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version=APP_VERSION,
    )


@app.get("/version", response_model=VersionResponse, tags=["Ops"])
def version() -> VersionResponse:
    """Return version information for the API and model."""
    return VersionResponse(
        version=APP_VERSION,
        model_version=MODEL_VERSION,
        python_version=sys.version.split()[0],
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Ops"])
def metrics() -> MetricsResponse:
    """Return the model's performance metrics from the last training run."""
    m = _state.get("metrics", {})
    if not m:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return MetricsResponse(
        train_mae=m["train_mae"],
        test_mae=m["test_mae"],
        test_rmse=m["test_rmse"],
        test_r2=m["test_r2"],
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_single(body: PredictRequest) -> PredictResponse:
    """Predict the annual insurance charge for a single individual."""
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        X = _build_feature_df(body)
        charge = float(predict(model, X)[0])
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction error") from exc
    return PredictResponse(predicted_charge=max(0.0, charge), model_version=MODEL_VERSION)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(body: BatchPredictRequest) -> BatchPredictResponse:
    """Predict insurance charges for a batch of up to 1000 individuals."""
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        predictions = [
            max(0.0, float(predict(model, _build_feature_df(r))[0]))
            for r in body.records
        ]
    except Exception as exc:
        logger.error("Batch prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Batch prediction error") from exc
    return BatchPredictResponse(
        predictions=predictions,
        count=len(predictions),
        model_version=MODEL_VERSION,
    )
