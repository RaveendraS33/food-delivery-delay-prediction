"""Prometheus metrics for the prediction API."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "delivery_requests_total",
    "Total API requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "delivery_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

PREDICTED_ETA = Histogram(
    "delivery_predicted_eta_minutes",
    "Distribution of predicted ETA (minutes)",
    buckets=(10, 15, 20, 25, 30, 40, 50, 60, 90),
)

DELAY_PROBABILITY = Histogram(
    "delivery_delay_probability",
    "Distribution of predicted delay probability",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

HIGH_RISK_PREDICTIONS = Counter(
    "delivery_high_risk_predictions_total",
    "Predictions flagged as high delay risk",
)

MODEL_LOADED = Gauge(
    "delivery_model_loaded",
    "1 if a trained model bundle is loaded, else 0",
)
