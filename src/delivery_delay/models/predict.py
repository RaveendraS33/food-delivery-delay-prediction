"""Inference: ETA + delay probability for an order, with parity to training.

A prediction starts from a high-level request (restaurant + customer location,
when, optionally traffic/vehicle/kitchen load). ``build_order_context`` turns
that into a one-row canonical frame -- filling weather from Open-Meteo, a prep
estimate, and a promised-time prior -- and ``DelayPredictor`` runs it through
the *same* ``build_features`` used in training.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from delivery_delay.config import Config, load_config
from delivery_delay.data.weather import WeatherClient, default_client
from delivery_delay.features.build import build_features
from delivery_delay.features.geo import haversine_km
from delivery_delay.models.registry import ModelBundle, load_bundle

logger = logging.getLogger(__name__)


def _infer_traffic_level(hour: int, cfg: Config) -> str:
    lunch = cfg.get("features.lunch_peak", [11, 14])
    dinner = cfg.get("features.dinner_peak", [17, 21])
    if dinner[0] <= hour < dinner[1]:
        return "high"
    if lunch[0] <= hour < lunch[1]:
        return "medium"
    if hour >= 22 or hour < 5:
        return "low"
    return "medium"


def build_order_context(
    restaurant_lat: float,
    restaurant_lon: float,
    customer_lat: float,
    customer_lon: float,
    order_time: datetime | None = None,
    traffic_level: str | None = None,
    vehicle_type: str = "scooter",
    active_orders: int | None = None,
    prep_time_minutes: float | None = None,
    cfg: Config | None = None,
    weather_client: WeatherClient | None = None,
) -> pd.DataFrame:
    """Assemble a one-row canonical-schema frame for a prediction request."""
    cfg = cfg or load_config()
    weather_client = weather_client or default_client
    order_time = order_time or datetime.now()
    hour = order_time.hour

    distance_km = haversine_km(restaurant_lat, restaurant_lon, customer_lat, customer_lon)

    if traffic_level is None:
        traffic_level = _infer_traffic_level(hour, cfg)
    if active_orders is None:
        active_orders = 8 if _infer_traffic_level(hour, cfg) in ("high", "jam") else 4
    if prep_time_minutes is None:
        prep_time_minutes = float(cfg.get("generator.base_prep_minutes", 12)) + 2.0

    weather = weather_client.weather_at(
        restaurant_lat, restaurant_lon, iso_hour=order_time.strftime("%Y-%m-%dT%H:00")
    )

    avg_speed = float(cfg.get("features.avg_speed_kmph", 22.0))
    promise_buffer = float(cfg.get("generator.promise_buffer_minutes", 10))
    nominal_travel = distance_km / avg_speed * 60.0
    promised = prep_time_minutes + nominal_travel + promise_buffer

    return pd.DataFrame(
        [
            {
                "order_id": "REQ",
                "timestamp": pd.Timestamp(order_time),
                "restaurant_id": "R-REQ",
                "restaurant_lat": restaurant_lat,
                "restaurant_lon": restaurant_lon,
                "customer_lat": customer_lat,
                "customer_lon": customer_lon,
                "prep_time_minutes": prep_time_minutes,
                "traffic_level": traffic_level,
                "vehicle_type": vehicle_type,
                "active_orders": active_orders,
                "weather_temp_c": weather["weather_temp_c"],
                "weather_precip_mm": weather["weather_precip_mm"],
                "weather_wind_kmph": weather["weather_wind_kmph"],
                "promised_minutes": round(promised, 2),
                "actual_minutes": float("nan"),  # unknown at request time
            }
        ]
    )


class DelayPredictor:
    """Loads a trained bundle and scores canonical-schema rows."""

    def __init__(self, model_dir: str | Path | None = None, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.bundle: ModelBundle = load_bundle(model_dir or self.cfg.model_dir)
        logger.info(
            "Loaded model bundle: %d features, trained %s (source=%s)",
            len(self.bundle.feature_columns),
            self.bundle.trained_at or "unknown",
            self.bundle.source,
        )

    def _risk_level(self, prob: float) -> str:
        high = self.bundle.high_risk_threshold
        if prob >= high:
            return "high"
        if prob >= high / 2:
            return "medium"
        return "low"

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a canonical-schema frame; returns one row of predictions per input row."""
        X = build_features(df, self.cfg).reindex(
            columns=self.bundle.feature_columns, fill_value=0.0
        )
        # Cast to float64 so JSON serialises clean values (XGBoost returns float32).
        eta = self.bundle.eta_model.predict(X).astype("float64")
        prob = self.bundle.delay_model.predict_proba(X)[:, 1].astype("float64")

        promised = pd.to_numeric(df["promised_minutes"], errors="coerce").to_numpy()
        result = pd.DataFrame(
            {
                "predicted_eta_minutes": eta.round(1),
                "promised_minutes": promised.round(1),
                "predicted_delay_minutes": (eta - promised).round(1),
                "delay_probability": prob.round(4),
                "on_time_probability": (1 - prob).round(4),
            }
        )
        result["risk_level"] = [self._risk_level(p) for p in prob]
        return result

    def predict_context(self, **kwargs) -> dict:
        """Convenience: build a context from kwargs and return a single prediction dict."""
        ctx = build_order_context(cfg=self.cfg, **kwargs)
        row = self.predict_frame(ctx).iloc[0].to_dict()
        # surface a couple of context fields useful to callers
        row["distance_km"] = round(
            float(
                haversine_km(
                    ctx.at[0, "restaurant_lat"],
                    ctx.at[0, "restaurant_lon"],
                    ctx.at[0, "customer_lat"],
                    ctx.at[0, "customer_lon"],
                )
            ),
            2,
        )
        row["weather_precip_mm"] = float(ctx.at[0, "weather_precip_mm"])
        return row
