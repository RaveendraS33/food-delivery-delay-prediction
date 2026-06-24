"""Load training data in the canonical schema.

Three sources are supported:

* ``synthetic`` -- generate everything offline (always available).
* ``public``    -- read a Kaggle-style food-delivery CSV from ``data/raw`` and
                   map it into the canonical schema.
* ``hybrid``    -- (default) use the public dataset if it is present and append
                   synthetic orders for volume + the live-event demo; otherwise
                   fall back to synthetic only.

The public path is best-effort: column names vary between Kaggle uploads, so we
map the well-known "Zomato/Swiggy delivery time" schema and fill anything that
is missing with neutral defaults or a derived prior. See ``data/README.md`` for
the expected file. Whatever the source, the returned frame goes through
``add_targets`` so ``actual_minutes``/``promised_minutes`` -> delay label stays
consistent.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from delivery_delay.config import Config, load_config
from delivery_delay.data.generator import (
    CANONICAL_COLUMNS,
    add_targets,
    generate_orders,
)

logger = logging.getLogger(__name__)

# Candidate filenames we look for in data/raw for the public dataset.
PUBLIC_CSV_CANDIDATES = ["orders.csv", "zomato_delivery.csv", "deliverytime.csv"]

# Map common Kaggle weather/traffic strings onto our numeric/categorical fields.
_TRAFFIC_MAP = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "jam": "jam",
}
_WEATHER_PRECIP = {  # crude precip proxy (mm) from condition strings
    "sunny": 0.0,
    "clear": 0.0,
    "cloudy": 0.0,
    "windy": 0.0,
    "fog": 0.5,
    "stormy": 6.0,
    "sandstorms": 1.0,
    "rain": 4.0,
    "rainy": 4.0,
}


def _find_public_csv(cfg: Config) -> Path | None:
    raw_dir = cfg.path("data_raw")
    for name in PUBLIC_CSV_CANDIDATES:
        candidate = raw_dir / name
        if candidate.exists():
            return candidate
    return None


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        c.strip().lower().replace("(", "_").replace(")", "").replace(" ", "_") for c in df.columns
    ]
    return df


def map_public_frame(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Best-effort map a Kaggle delivery dataframe into the canonical schema."""
    df = _norm_cols(df)
    n = len(df)
    rng = np.random.default_rng(cfg.seed)
    avg_speed = float(cfg.get("features.avg_speed_kmph", 22.0))
    base_prep = float(cfg.get("generator.base_prep_minutes", 12))
    promise_buffer = float(cfg.get("generator.promise_buffer_minutes", 10))

    def col(*names, default=None):
        for name in names:
            if name in df.columns:
                return df[name]
        return pd.Series([default] * n)

    r_lat = pd.to_numeric(col("restaurant_latitude", default=np.nan), errors="coerce")
    r_lon = pd.to_numeric(col("restaurant_longitude", default=np.nan), errors="coerce")
    c_lat = pd.to_numeric(col("delivery_location_latitude", default=np.nan), errors="coerce")
    c_lon = pd.to_numeric(col("delivery_location_longitude", default=np.nan), errors="coerce")

    actual = pd.to_numeric(col("time_taken_min", "time_taken", default=np.nan), errors="coerce")

    # Timestamp: combine order date + order time when available, else synthesize.
    ts = pd.to_datetime(
        col("order_date").astype(str) + " " + col("time_orderd").astype(str),
        errors="coerce",
        dayfirst=True,
    )
    if ts.isna().all():
        base = pd.Timestamp("2025-01-01")
        ts = pd.Series([base + pd.Timedelta(hours=int(h)) for h in rng.integers(8, 23, n)])

    traffic_raw = col("road_traffic_density", default="medium").astype(str).str.strip().str.lower()
    traffic = traffic_raw.map(_TRAFFIC_MAP).fillna("medium")

    weather_raw = (
        col("weatherconditions", "weather_conditions", default="sunny")
        .astype(str)
        .str.replace("conditions", "", regex=False)
        .str.strip()
        .str.lower()
    )
    precip = weather_raw.map(_WEATHER_PRECIP).fillna(0.0)

    vehicle_raw = col("type_of_vehicle", default="scooter").astype(str).str.strip().str.lower()
    vehicle = vehicle_raw.replace(
        {"motorcycle": "scooter", "electric_scooter": "scooter", "bicycle": "bike"}
    )
    vehicle = vehicle.where(vehicle.isin(["bike", "scooter", "car"]), "scooter")

    multiple = pd.to_numeric(col("multiple_deliveries", default=0), errors="coerce").fillna(0)
    active_orders = (2 + 3 * multiple).astype(int)

    out = pd.DataFrame(
        {
            "order_id": col("id", default=None).fillna(
                pd.Series([f"PUB-{i:07d}" for i in range(n)])
            ),
            "timestamp": ts,
            "restaurant_id": col("delivery_person_id", default="R-PUB").astype(str),
            "restaurant_lat": r_lat,
            "restaurant_lon": r_lon,
            "customer_lat": c_lat,
            "customer_lon": c_lon,
            "prep_time_minutes": base_prep,
            "traffic_level": traffic,
            "vehicle_type": vehicle,
            "active_orders": active_orders,
            "weather_temp_c": 18.0,
            "weather_precip_mm": precip,
            "weather_wind_kmph": 10.0,
            "actual_minutes": actual,
        }
    )

    # Derive a promised quote consistent with the synthetic generator so the
    # delay label means the same thing across sources.
    from delivery_delay.features.geo import haversine_km

    dist = haversine_km(
        out["restaurant_lat"], out["restaurant_lon"], out["customer_lat"], out["customer_lon"]
    )
    nominal_travel = (dist / avg_speed * 60.0).fillna(15.0)
    out["promised_minutes"] = (base_prep + nominal_travel + promise_buffer).round(2)

    # Drop rows with no usable target or coordinates.
    out = out.dropna(subset=["actual_minutes", "restaurant_lat", "customer_lat"])
    out = out[out["actual_minutes"] > 0]
    out["order_id"] = out["order_id"].astype(str)
    return out[CANONICAL_COLUMNS].reset_index(drop=True)


def load_public(cfg: Config) -> pd.DataFrame | None:
    """Load + map the public dataset if a recognised CSV is present."""
    path = _find_public_csv(cfg)
    if path is None:
        return None
    try:
        raw = pd.read_csv(path)
        mapped = map_public_frame(raw, cfg)
        logger.info("Loaded %d rows from public dataset %s", len(mapped), path.name)
        return mapped
    except Exception as exc:  # pragma: no cover - depends on external file
        logger.warning("Could not map public dataset %s: %s", path, exc)
        return None


def load_canonical(
    cfg: Config | None = None,
    source: str = "hybrid",
    n_orders: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Return a canonical-schema training frame *with targets* for a source."""
    cfg = cfg or load_config()

    if source == "synthetic":
        frame = generate_orders(cfg, n_orders=n_orders, seed=seed)
    elif source == "public":
        public = load_public(cfg)
        if public is None or public.empty:
            raise FileNotFoundError(
                "source='public' but no usable CSV found in data/raw "
                f"(looked for {PUBLIC_CSV_CANDIDATES}). See data/README.md."
            )
        frame = public
    elif source == "hybrid":
        public = load_public(cfg)
        synth = generate_orders(cfg, n_orders=n_orders, seed=seed)
        if public is not None and not public.empty:
            frame = pd.concat([public, synth], ignore_index=True)
            logger.info("Hybrid source: %d public + %d synthetic rows", len(public), len(synth))
        else:
            logger.info("No public dataset found; using synthetic only.")
            frame = synth
    else:
        raise ValueError(f"Unknown source: {source!r} (use synthetic|public|hybrid)")

    return add_targets(frame, cfg)
