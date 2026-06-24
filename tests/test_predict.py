from datetime import datetime

from delivery_delay.recommend import recommend_order_time


def test_predict_context_ranges(predictor):
    out = predictor.predict_context(
        restaurant_lat=42.36,
        restaurant_lon=-71.06,
        customer_lat=42.35,
        customer_lon=-71.08,
        order_time=datetime(2025, 6, 7, 19, 0),
        vehicle_type="scooter",
        active_orders=10,
    )
    assert out["predicted_eta_minutes"] > 0
    assert 0.0 <= out["delay_probability"] <= 1.0
    assert abs(out["delay_probability"] + out["on_time_probability"] - 1.0) < 1e-6
    assert out["risk_level"] in {"low", "medium", "high"}
    assert out["distance_km"] >= 0


def test_longer_distance_increases_eta(predictor):
    near = predictor.predict_context(
        restaurant_lat=42.360,
        restaurant_lon=-71.060,
        customer_lat=42.362,
        customer_lon=-71.062,
        order_time=datetime(2025, 6, 3, 13, 0),
    )
    far = predictor.predict_context(
        restaurant_lat=42.360,
        restaurant_lon=-71.060,
        customer_lat=42.390,
        customer_lon=-71.110,
        order_time=datetime(2025, 6, 3, 13, 0),
    )
    assert far["predicted_eta_minutes"] > near["predicted_eta_minutes"]


def test_recommend_schedule_shape(predictor):
    rec = recommend_order_time(
        predictor,
        restaurant_lat=42.36,
        restaurant_lon=-71.06,
        customer_lat=42.35,
        customer_lon=-71.08,
        start_time=datetime(2025, 6, 7, 18, 0),
        lookahead_hours=5,
    )
    assert len(rec["schedule"]) == 6
    assert rec["best"]["delay_probability"] <= rec["current"]["delay_probability"]
    assert isinstance(rec["advice"], str) and rec["advice"]
