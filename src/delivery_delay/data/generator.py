"""Synthetic food-delivery event generator.

Produces orders in the project's *canonical schema* (the same schema the public
dataset loader maps into, and the same one the feature pipeline consumes). The
generator bakes in realistic, learnable relationships:

* longer distance            -> longer ETA
* lunch/dinner peak hours    -> more load, slower traffic, more delay
* rain / high wind           -> slower travel, more delay
* busy kitchen (active_orders) -> longer prep
* heavy traffic              -> slower travel

Generation is fully offline and deterministic for a given seed, so training
runs and tests are reproducible. (Live weather from Open-Meteo is only used on
the serving path -- see ``weather.py``.)
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from delivery_delay.config import Config, load_config

# Canonical column order produced by every data source in this project.
CANONICAL_COLUMNS = [
    "order_id",
    "timestamp",
    "restaurant_id",
    "restaurant_lat",
    "restaurant_lon",
    "customer_lat",
    "customer_lon",
    "prep_time_minutes",
    "traffic_level",
    "vehicle_type",
    "active_orders",
    "weather_temp_c",
    "weather_precip_mm",
    "weather_wind_kmph",
    "promised_minutes",
    "actual_minutes",
]

TRAFFIC_LEVELS = ["low", "medium", "high", "jam"]
VEHICLE_TYPES = ["bike", "scooter", "car"]
VEHICLE_SPEED_FACTOR = {"bike": 0.85, "scooter": 1.0, "car": 1.05}

# Reference date the simulated window ends on. Fixed for determinism; only the
# hour-of-day / day-of-week derived from it matter to the model.
_REFERENCE_DATE = datetime(2025, 6, 1)
_WINDOW_DAYS = 120


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorised great-circle distance in km."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _hour_traffic_multiplier(hours: np.ndarray, cfg: Config) -> np.ndarray:
    """Smooth traffic multiplier peaking during lunch and dinner windows."""
    lunch = cfg.get("features.lunch_peak", [11, 14])
    dinner = cfg.get("features.dinner_peak", [17, 21])
    lunch_mid = (lunch[0] + lunch[1]) / 2
    dinner_mid = (dinner[0] + dinner[1]) / 2

    base = 1.0
    lunch_bump = 0.45 * np.exp(-((hours - lunch_mid) ** 2) / 3.0)
    dinner_bump = 0.65 * np.exp(-((hours - dinner_mid) ** 2) / 4.0)
    # quieter overnight
    night_dip = -0.2 * np.exp(-((hours - 3) ** 2) / 6.0)
    return base + lunch_bump + dinner_bump + night_dip


def _traffic_label(mult: np.ndarray) -> np.ndarray:
    labels = np.where(
        mult < 1.15,
        "low",
        np.where(mult < 1.4, "medium", np.where(mult < 1.65, "high", "jam")),
    )
    return labels


def generate_orders(
    cfg: Config | None = None,
    n_orders: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a canonical-schema DataFrame of synthetic delivery orders."""
    cfg = cfg or load_config()
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)

    n_orders = int(n_orders or cfg.get("generator.n_orders", 40000))
    n_rest = int(cfg.get("generator.n_restaurants", 80))
    base_prep = float(cfg.get("generator.base_prep_minutes", 12))
    promise_buffer = float(cfg.get("generator.promise_buffer_minutes", 10))
    avg_speed = float(cfg.get("features.avg_speed_kmph", 22.0))

    lat_min = cfg.get("city.lat_min", 42.31)
    lat_max = cfg.get("city.lat_max", 42.39)
    lon_min = cfg.get("city.lon_min", -71.12)
    lon_max = cfg.get("city.lon_max", -71.03)

    # --- Restaurants ----------------------------------------------------
    rest_lat = rng.uniform(lat_min, lat_max, n_rest)
    rest_lon = rng.uniform(lon_min, lon_max, n_rest)
    rest_base_prep = base_prep + rng.gamma(shape=2.0, scale=2.5, size=n_rest)
    rest_popularity = rng.gamma(shape=2.0, scale=1.0, size=n_rest)
    rest_popularity /= rest_popularity.sum()

    # --- Orders: choose restaurants weighted by popularity --------------
    r_idx = rng.choice(n_rest, size=n_orders, p=rest_popularity)
    r_lat = rest_lat[r_idx]
    r_lon = rest_lon[r_idx]

    # Customers scattered near their restaurant (degrees ~ a few km).
    c_lat = np.clip(r_lat + rng.normal(0, 0.020, n_orders), lat_min, lat_max)
    c_lon = np.clip(r_lon + rng.normal(0, 0.026, n_orders), lon_min, lon_max)
    distance_km = _haversine_km(r_lat, r_lon, c_lat, c_lon)

    # --- Timestamps weighted toward meal peaks --------------------------
    hour_weights = _hour_traffic_multiplier(np.arange(24), cfg)
    hour_weights = hour_weights / hour_weights.sum()
    hours = rng.choice(24, size=n_orders, p=hour_weights)
    minutes = rng.integers(0, 60, n_orders)
    day_offsets = rng.integers(0, _WINDOW_DAYS, n_orders)
    start = _REFERENCE_DATE - timedelta(days=_WINDOW_DAYS)
    timestamps = [
        start + timedelta(days=int(d), hours=int(h), minutes=int(m))
        for d, h, m in zip(day_offsets, hours, minutes, strict=True)
    ]

    # --- Traffic --------------------------------------------------------
    traffic_mult = _hour_traffic_multiplier(hours.astype(float), cfg)
    traffic_mult = traffic_mult * rng.normal(1.0, 0.08, n_orders)
    traffic_mult = np.clip(traffic_mult, 0.7, 2.2)
    traffic_level = _traffic_label(traffic_mult)

    # --- Kitchen load (busier at peaks; popular spots get hit harder) ---
    load_lambda = 4.0 * traffic_mult * (1 + rest_popularity[r_idx] * n_rest * 0.06)
    active_orders = rng.poisson(np.clip(load_lambda, 0.5, 28))

    # --- Weather (synthetic, offline) -----------------------------------
    # Temperature varies by hour around a seasonal mean; occasional rain.
    temp_c = 19 + 6 * np.sin((hours - 9) / 24 * 2 * np.pi) + rng.normal(0, 2.5, n_orders)
    rain_event = rng.random(n_orders) < 0.16
    precip_mm = np.where(rain_event, rng.exponential(2.5, n_orders), 0.0)
    wind_kmph = rng.gamma(shape=2.0, scale=4.5, size=n_orders)

    # --- Vehicle --------------------------------------------------------
    vehicle = rng.choice(VEHICLE_TYPES, size=n_orders, p=[0.45, 0.35, 0.20])
    veh_factor = np.array([VEHICLE_SPEED_FACTOR[v] for v in vehicle])

    # --- Prep time (surges when the kitchen is busy) --------------------
    prep = rest_base_prep[r_idx] * (1 + 0.07 * active_orders)
    prep += rng.normal(0, 2.0, n_orders)
    prep = np.clip(prep, 4, None)

    # --- Travel time (slower in traffic / bad weather) ------------------
    weather_slow = 1 + 0.05 * precip_mm + 0.005 * wind_kmph
    effective_speed = avg_speed * veh_factor / (traffic_mult * weather_slow)
    travel = distance_km / np.clip(effective_speed, 4, None) * 60.0  # minutes

    # --- Actual delivery time -------------------------------------------
    weather_penalty = 1.6 * precip_mm + 0.06 * wind_kmph
    actual = prep + travel + weather_penalty + rng.normal(0, 2.5, n_orders)
    actual = np.clip(actual, 6, None)

    # --- Promised quote (deliberately a bit optimistic) -----------------
    nominal_speed = avg_speed * veh_factor
    nominal_travel = distance_km / nominal_speed * 60.0
    promised = rest_base_prep[r_idx] + nominal_travel + promise_buffer
    promised = np.clip(promised, 10, None)

    df = pd.DataFrame(
        {
            "order_id": [f"ORD-{i:07d}" for i in range(n_orders)],
            "timestamp": pd.to_datetime(timestamps),
            "restaurant_id": [f"R-{i:04d}" for i in r_idx],
            "restaurant_lat": r_lat,
            "restaurant_lon": r_lon,
            "customer_lat": c_lat,
            "customer_lon": c_lon,
            "prep_time_minutes": np.round(prep, 2),
            "traffic_level": traffic_level,
            "vehicle_type": vehicle,
            "active_orders": active_orders.astype(int),
            "weather_temp_c": np.round(temp_c, 2),
            "weather_precip_mm": np.round(precip_mm, 2),
            "weather_wind_kmph": np.round(wind_kmph, 2),
            "promised_minutes": np.round(promised, 2),
            "actual_minutes": np.round(actual, 2),
        }
    )
    return df[CANONICAL_COLUMNS]


def add_targets(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Append the supervised targets derived from promised vs actual time."""
    cfg = cfg or load_config()
    threshold = float(cfg.get("generator.delay_threshold_minutes", 10))
    out = df.copy()
    out["delay_minutes"] = out["actual_minutes"] - out["promised_minutes"]
    out["is_delayed"] = (out["delay_minutes"] > threshold).astype(int)
    return out


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    frame = add_targets(generate_orders(n_orders=2000))
    print(frame.head())
    print(f"\nrows={len(frame)}  delay_rate={frame['is_delayed'].mean():.3f}")
