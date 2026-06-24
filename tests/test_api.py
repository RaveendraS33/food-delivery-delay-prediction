import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(trained_model_dir):
    # trained_model_dir sets MODEL_DIR; import app afterwards so lifespan finds it.
    from delivery_delay.api.main import app

    with TestClient(app) as c:
        yield c


def _order():
    return {
        "restaurant_lat": 42.36,
        "restaurant_lon": -71.06,
        "customer_lat": 42.35,
        "customer_lon": -71.08,
        "order_time": "2025-06-07T19:00:00",
        "vehicle_type": "scooter",
        "active_orders": 8,
    }


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["loaded"] is True
    assert "delay_roc_auc" in body["metrics"]


def test_predict(client):
    r = client.post("/predict", json=_order())
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_eta_minutes"] > 0
    assert 0.0 <= body["delay_probability"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high"}


def test_predict_validation_error(client):
    bad = _order()
    bad["restaurant_lat"] = 999  # out of range
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_recommend(client):
    payload = _order()
    payload["start_time"] = payload.pop("order_time")
    payload["lookahead_hours"] = 4
    r = client.post("/recommend", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body["schedule"]) == 5
    assert "advice" in body


def test_metrics_endpoint(client):
    client.post("/predict", json=_order())
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "delivery_requests_total" in r.text
