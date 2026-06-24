"""Temporal features derived from the order timestamp.

Returns a DataFrame so the same logic serves both batch training (a column of
timestamps) and single-request serving (a one-row frame).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from delivery_delay.config import Config, load_config

MEAL_PERIODS = ["breakfast", "lunch", "dinner", "late", "off_peak"]


def meal_period(hour: int, cfg: Config) -> str:
    lunch = cfg.get("features.lunch_peak", [11, 14])
    dinner = cfg.get("features.dinner_peak", [17, 21])
    if 6 <= hour < 10:
        return "breakfast"
    if lunch[0] <= hour < lunch[1]:
        return "lunch"
    if dinner[0] <= hour < dinner[1]:
        return "dinner"
    if hour >= 22 or hour < 4:
        return "late"
    return "off_peak"


def temporal_features(timestamps: pd.Series, cfg: Config | None = None) -> pd.DataFrame:
    """Build temporal features from a Series of datetimes."""
    cfg = cfg or load_config()
    ts = pd.to_datetime(timestamps)

    hour = ts.dt.hour
    dow = ts.dt.dayofweek  # Mon=0 .. Sun=6

    lunch = cfg.get("features.lunch_peak", [11, 14])
    dinner = cfg.get("features.dinner_peak", [17, 21])

    is_lunch_peak = ((hour >= lunch[0]) & (hour < lunch[1])).astype(int)
    is_dinner_peak = ((hour >= dinner[0]) & (hour < dinner[1])).astype(int)

    out = pd.DataFrame(
        {
            "hour": hour.astype(int),
            "day_of_week": dow.astype(int),
            "is_weekend": (dow >= 5).astype(int),
            "is_lunch_peak": is_lunch_peak,
            "is_dinner_peak": is_dinner_peak,
            "is_peak": ((is_lunch_peak == 1) | (is_dinner_peak == 1)).astype(int),
            # cyclical encodings so the model sees 23:00 and 00:00 as adjacent
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
        },
        index=ts.index,
    )
    out["meal_period"] = [meal_period(int(h), cfg) for h in hour]
    return out
