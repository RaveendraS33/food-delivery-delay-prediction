"""Feature assembly.

``build_features`` turns a canonical-schema frame (one or many rows) into a
purely numeric feature matrix. The *exact same* function runs at training time
and at serving time, which guarantees train/serve parity. Categorical fields
are one-hot encoded against fixed vocabularies so the column set never depends
on which categories happen to appear in a given batch -- a single prediction
request produces the same columns as the full training set.
"""

from __future__ import annotations

import pandas as pd

from delivery_delay.config import Config, load_config
from delivery_delay.features.geo import haversine_km
from delivery_delay.features.temporal import MEAL_PERIODS, temporal_features

# Fixed categorical vocabularies -> stable one-hot columns.
TRAFFIC_CATEGORIES = ["low", "medium", "high", "jam"]
VEHICLE_CATEGORIES = ["bike", "scooter", "car"]
MEAL_CATEGORIES = MEAL_PERIODS

# Numeric features taken/derived directly from the canonical frame.
_BASE_NUMERIC = [
    "distance_km",
    "prep_time_minutes",
    "active_orders",
    "weather_temp_c",
    "weather_precip_mm",
    "weather_wind_kmph",
    "promised_minutes",
]

TARGET_ETA = "actual_minutes"
TARGET_DELAY = "is_delayed"


def _one_hot(series: pd.Series, prefix: str, categories: list[str]) -> pd.DataFrame:
    cat = pd.Categorical(series.astype(str).str.lower(), categories=categories)
    dummies = pd.get_dummies(cat, prefix=prefix)
    # Ensure every category column exists even if absent in this batch.
    for c in categories:
        col = f"{prefix}_{c}"
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[[f"{prefix}_{c}" for c in categories]]
    dummies.index = series.index
    return dummies.astype(int)


def build_features(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Build the numeric feature matrix for a canonical-schema frame."""
    cfg = cfg or load_config()
    df = df.reset_index(drop=True)

    feats = pd.DataFrame(index=df.index)
    feats["distance_km"] = haversine_km(
        df["restaurant_lat"], df["restaurant_lon"], df["customer_lat"], df["customer_lon"]
    )
    for col in _BASE_NUMERIC:
        if col == "distance_km":
            continue
        feats[col] = pd.to_numeric(df[col], errors="coerce")

    # Temporal block
    temp = temporal_features(df["timestamp"], cfg)
    meal = temp.pop("meal_period")
    feats = pd.concat([feats, temp], axis=1)

    # Categorical one-hot blocks
    feats = pd.concat(
        [
            feats,
            _one_hot(df["traffic_level"], "traffic", TRAFFIC_CATEGORIES),
            _one_hot(df["vehicle_type"], "vehicle", VEHICLE_CATEGORIES),
            _one_hot(meal, "meal", MEAL_CATEGORIES),
        ],
        axis=1,
    )

    # A couple of cheap interaction priors XGBoost can exploit.
    feats["load_x_peak"] = feats["active_orders"] * feats["is_peak"]
    feats["dist_x_rain"] = feats["distance_km"] * feats["weather_precip_mm"]

    return feats.fillna(0.0)


def feature_columns(cfg: Config | None = None) -> list[str]:
    """The canonical, ordered list of feature columns produced by build_features."""
    cfg = cfg or load_config()
    cols = list(_BASE_NUMERIC)
    cols += [
        "hour",
        "day_of_week",
        "is_weekend",
        "is_lunch_peak",
        "is_dinner_peak",
        "is_peak",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
    cols += [f"traffic_{c}" for c in TRAFFIC_CATEGORIES]
    cols += [f"vehicle_{c}" for c in VEHICLE_CATEGORIES]
    cols += [f"meal_{c}" for c in MEAL_CATEGORIES]
    cols += ["load_x_peak", "dist_x_rain"]
    return cols


def build_xy(df: pd.DataFrame, cfg: Config | None = None):
    """Return (X, y_eta, y_delay) from a canonical frame *with targets*."""
    cfg = cfg or load_config()
    X = build_features(df, cfg)
    X = X.reindex(columns=feature_columns(cfg), fill_value=0.0)
    y_eta = pd.to_numeric(df[TARGET_ETA], errors="coerce")
    y_delay = df[TARGET_DELAY].astype(int)
    return X, y_eta, y_delay
