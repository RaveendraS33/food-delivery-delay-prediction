"""FastAPI prediction service.

Endpoints:
    GET  /            -- service banner + links
    GET  /health      -- liveness + whether a model is loaded
    GET  /model/info  -- training metadata + metrics
    POST /predict     -- ETA + delay probability for one order
    POST /recommend   -- best time to order within a lookahead window
    GET  /metrics     -- Prometheus exposition

The trained model bundle is loaded once at startup. If no model is present the
service still starts and prediction endpoints return 503 with guidance to run
training first.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from delivery_delay import __version__
from delivery_delay.api import metrics as M
from delivery_delay.api.schemas import (
    ModelInfo,
    PredictRequest,
    PredictResponse,
    RecommendRequest,
    RecommendResponse,
)
from delivery_delay.config import load_config
from delivery_delay.models.predict import DelayPredictor
from delivery_delay.recommend import recommend_order_time

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    app.state.cfg = cfg
    try:
        app.state.predictor = DelayPredictor(cfg=cfg)
        M.MODEL_LOADED.set(1)
        logger.info("Model bundle loaded.")
    except FileNotFoundError as exc:
        app.state.predictor = None
        M.MODEL_LOADED.set(0)
        logger.warning("Starting without a model: %s", exc)
    yield


app = FastAPI(
    title="Food Delivery Delay Prediction API",
    description="ETA regression + delay-probability classification with an optimal-ordering recommender.",
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    endpoint = request.url.path
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        M.REQUEST_COUNT.labels(endpoint=endpoint, status=500).inc()
        raise
    elapsed = time.perf_counter() - start
    M.REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
    M.REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()
    return response


def _get_predictor(request: Request) -> DelayPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model loaded. Run `python scripts/train.py` and restart the API.",
        )
    return predictor


@app.get("/")
def root():
    return {
        "service": "food-delivery-delay-prediction",
        "version": __version__,
        "docs": "/docs",
        "endpoints": ["/health", "/model/info", "/predict", "/recommend", "/metrics"],
    }


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": getattr(request.app.state, "predictor", None) is not None,
    }


@app.get("/model/info", response_model=ModelInfo)
def model_info(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        return ModelInfo(loaded=False)
    b = predictor.bundle
    return ModelInfo(
        loaded=True,
        trained_at=b.trained_at,
        source=b.source,
        n_train_rows=b.n_train_rows,
        n_features=len(b.feature_columns),
        metrics=b.metrics,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    predictor = _get_predictor(request)
    order_time = req.order_time.replace(tzinfo=None) if req.order_time else None
    result = predictor.predict_context(
        restaurant_lat=req.restaurant_lat,
        restaurant_lon=req.restaurant_lon,
        customer_lat=req.customer_lat,
        customer_lon=req.customer_lon,
        order_time=order_time,
        traffic_level=req.traffic_level,
        vehicle_type=req.vehicle_type,
        active_orders=req.active_orders,
        prep_time_minutes=req.prep_time_minutes,
    )

    M.PREDICTED_ETA.observe(result["predicted_eta_minutes"])
    M.DELAY_PROBABILITY.observe(result["delay_probability"])
    if result["risk_level"] == "high":
        M.HIGH_RISK_PREDICTIONS.inc()

    return PredictResponse(**result)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest, request: Request):
    predictor = _get_predictor(request)
    start_time = req.start_time.replace(tzinfo=None) if req.start_time else None
    result = recommend_order_time(
        predictor,
        restaurant_lat=req.restaurant_lat,
        restaurant_lon=req.restaurant_lon,
        customer_lat=req.customer_lat,
        customer_lon=req.customer_lon,
        start_time=start_time,
        lookahead_hours=req.lookahead_hours,
        vehicle_type=req.vehicle_type,
        active_orders=req.active_orders,
        cfg=request.app.state.cfg,
    )
    return RecommendResponse(**result)


@app.get("/metrics")
def prometheus_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
