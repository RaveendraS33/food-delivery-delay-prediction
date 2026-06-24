import pandas as pd

from delivery_delay.features.temporal import meal_period, temporal_features


def test_temporal_columns_present(cfg):
    ts = pd.Series(pd.to_datetime(["2025-06-02 12:30", "2025-06-07 19:00"]))
    feats = temporal_features(ts, cfg)
    for col in ["hour", "day_of_week", "is_weekend", "is_peak", "hour_sin", "meal_period"]:
        assert col in feats.columns


def test_weekend_and_peak_flags(cfg):
    # 2025-06-02 is a Monday 12:30 (lunch); 2025-06-07 is a Saturday 19:00 (dinner).
    ts = pd.Series(pd.to_datetime(["2025-06-02 12:30", "2025-06-07 19:00"]))
    feats = temporal_features(ts, cfg)
    assert feats.iloc[0]["is_weekend"] == 0
    assert feats.iloc[0]["is_lunch_peak"] == 1
    assert feats.iloc[1]["is_weekend"] == 1
    assert feats.iloc[1]["is_dinner_peak"] == 1


def test_meal_period_buckets(cfg):
    assert meal_period(8, cfg) == "breakfast"
    assert meal_period(12, cfg) == "lunch"
    assert meal_period(19, cfg) == "dinner"
    assert meal_period(23, cfg) == "late"
    assert meal_period(15, cfg) == "off_peak"
