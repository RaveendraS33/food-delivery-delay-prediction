"""Streamlit decision-support dashboard.

Interactive UI delivering, for a chosen route and time:
  * predicted delivery ETA,
  * delay probability with a risk gauge,
  * a "best time to order" recommendation over the next few hours.

It prefers the FastAPI service (set DELIVERY_API_URL) and transparently falls
back to running the model in-process, so it works standalone after training.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Make the package importable for the in-process fallback path.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from delivery_delay.config import load_config  # noqa: E402

API_URL = os.getenv("DELIVERY_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Food Delivery Delay Predictor",
    page_icon="🛵",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Backend: try the API, fall back to an in-process predictor.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _local_predictor():
    from delivery_delay.models.predict import DelayPredictor

    return DelayPredictor()


def _api_up() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=1.5)
        return r.ok and r.json().get("model_loaded", False)
    except requests.RequestException:
        return False


def call_predict(payload: dict, use_api: bool) -> dict:
    if use_api:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=8)
        r.raise_for_status()
        return r.json()
    return _local_predictor().predict_context(
        restaurant_lat=payload["restaurant_lat"],
        restaurant_lon=payload["restaurant_lon"],
        customer_lat=payload["customer_lat"],
        customer_lon=payload["customer_lon"],
        order_time=(
            datetime.fromisoformat(payload["order_time"]) if payload.get("order_time") else None
        ),
        vehicle_type=payload.get("vehicle_type", "scooter"),
        active_orders=payload.get("active_orders"),
    )


def call_recommend(payload: dict, use_api: bool) -> dict:
    if use_api:
        r = requests.post(f"{API_URL}/recommend", json=payload, timeout=20)
        r.raise_for_status()
        return r.json()
    from delivery_delay.recommend import recommend_order_time

    return recommend_order_time(
        _local_predictor(),
        restaurant_lat=payload["restaurant_lat"],
        restaurant_lon=payload["restaurant_lon"],
        customer_lat=payload["customer_lat"],
        customer_lon=payload["customer_lon"],
        start_time=(
            datetime.fromisoformat(payload["start_time"]) if payload.get("start_time") else None
        ),
        lookahead_hours=payload.get("lookahead_hours"),
        vehicle_type=payload.get("vehicle_type", "scooter"),
        active_orders=payload.get("active_orders"),
    )


def model_metadata(use_api: bool) -> dict | None:
    if use_api:
        try:
            r = requests.get(f"{API_URL}/model/info", timeout=3)
            return r.json() if r.ok else None
        except requests.RequestException:
            return None
    try:
        b = _local_predictor().bundle
        return {
            "loaded": True,
            "trained_at": b.trained_at,
            "source": b.source,
            "n_train_rows": b.n_train_rows,
            "n_features": len(b.feature_columns),
            "metrics": b.metrics,
        }
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def risk_gauge(prob: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": "Delay risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f2937"},
                "steps": [
                    {"range": [0, 25], "color": "#86efac"},
                    {"range": [25, 50], "color": "#fde047"},
                    {"range": [50, 100], "color": "#fca5a5"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def schedule_chart(schedule: list[dict], best_offset: int) -> go.Figure:
    df = pd.DataFrame(schedule)
    colors = ["#2563eb" if o == best_offset else "#93c5fd" for o in df["offset_hours"]]
    fig = go.Figure(
        go.Bar(
            x=[s[11:16] for s in df["time"]],
            y=df["delay_probability"] * 100,
            marker_color=colors,
            text=[f"{p * 100:.0f}%" for p in df["delay_probability"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=320,
        title="Predicted delay risk by order time",
        xaxis_title="Order time",
        yaxis_title="Delay risk (%)",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# --------------------------------------------------------------------------- #
# Sidebar inputs
# --------------------------------------------------------------------------- #
cfg = load_config()
city_lat = (cfg.get("city.lat_min") + cfg.get("city.lat_max")) / 2
city_lon = (cfg.get("city.lon_min") + cfg.get("city.lon_max")) / 2

st.sidebar.title("🛵 Order details")
api_available = _api_up()
use_api = st.sidebar.toggle(
    "Use prediction API",
    value=api_available,
    help=f"API at {API_URL}. Off = run the model in-process.",
)
if use_api and not api_available:
    st.sidebar.warning("API not reachable — using in-process model instead.")
    use_api = False
st.sidebar.caption(f"Backend: {'FastAPI service' if use_api else 'in-process model'}")

st.sidebar.subheader("Restaurant")
r_lat = st.sidebar.number_input("Restaurant lat", value=round(city_lat + 0.01, 5), format="%.5f")
r_lon = st.sidebar.number_input("Restaurant lon", value=round(city_lon, 5), format="%.5f")

st.sidebar.subheader("Customer")
c_lat = st.sidebar.number_input("Customer lat", value=round(city_lat - 0.015, 5), format="%.5f")
c_lon = st.sidebar.number_input("Customer lon", value=round(city_lon + 0.01, 5), format="%.5f")

st.sidebar.subheader("Context")
vehicle = st.sidebar.selectbox("Vehicle", ["scooter", "bike", "car"])
active_orders = st.sidebar.slider("Kitchen load (active orders)", 0, 25, 6)
hour = st.sidebar.slider("Order hour", 0, 23, datetime.now().hour)
order_dt = datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("Food Delivery Delay Prediction")
st.caption(
    "Estimates delivery ETA and the probability an order is delivered late, then "
    "recommends the best time to order — using weather, distance, traffic, kitchen "
    "load, and time-of-day features."
)

payload = {
    "restaurant_lat": r_lat,
    "restaurant_lon": r_lon,
    "customer_lat": c_lat,
    "customer_lon": c_lon,
    "order_time": order_dt.isoformat(),
    "vehicle_type": vehicle,
    "active_orders": active_orders,
}

meta = model_metadata(use_api)
if not meta or not meta.get("loaded"):
    st.error(
        "No trained model found. Run `python scripts/train.py` first "
        "(and start the API if using the API backend)."
    )
    st.stop()

try:
    pred = call_predict(payload, use_api)
except Exception as exc:  # pragma: no cover - UI guard
    st.error(f"Prediction failed: {exc}")
    st.stop()

# --- Top-line metrics ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted ETA", f"{pred['predicted_eta_minutes']:.0f} min")
c2.metric("Promised", f"{pred['promised_minutes']:.0f} min")
delta = pred["predicted_delay_minutes"]
c3.metric("Predicted vs promised", f"{delta:+.0f} min", delta_color="inverse")
c4.metric("Distance", f"{pred['distance_km']:.1f} km")

left, right = st.columns([1, 1.3])
with left:
    risk_label = {"low": "🟢 Low", "medium": "🟡 Medium", "high": "🔴 High"}[pred["risk_level"]]
    st.subheader(f"Delay risk: {risk_label}")
    st.plotly_chart(risk_gauge(pred["delay_probability"]), use_container_width=True)
    if pred["weather_precip_mm"] > 0:
        st.info(f"🌧️ Forecast precipitation: {pred['weather_precip_mm']:.1f} mm")

with right:
    st.subheader("Route")
    points = pd.DataFrame(
        {
            "lat": [r_lat, c_lat],
            "lon": [r_lon, c_lon],
        }
    )
    st.map(points, size=40, zoom=12)

# --- Recommendation ---
st.divider()
st.subheader("⏱️ Best time to order")
lookahead = st.slider("Look ahead (hours)", 2, 12, int(cfg.get("recommend.lookahead_hours", 6)))
rec_payload = dict(payload)
rec_payload["start_time"] = order_dt.isoformat()
rec_payload["lookahead_hours"] = lookahead

try:
    rec = call_recommend(rec_payload, use_api)
    if rec["recommend_now"]:
        st.success(rec["advice"])
    else:
        st.warning(rec["advice"])
    st.plotly_chart(
        schedule_chart(rec["schedule"], rec["best"]["offset_hours"]),
        use_container_width=True,
    )
except Exception as exc:  # pragma: no cover - UI guard
    st.error(f"Recommendation failed: {exc}")

# --- Model card ---
with st.expander("Model details"):
    m = meta.get("metrics", {}) or {}
    st.write(
        {
            "trained_at": meta.get("trained_at"),
            "data_source": meta.get("source"),
            "train_rows": meta.get("n_train_rows"),
            "features": meta.get("n_features"),
        }
    )
    if m:
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("ETA MAE", f"{m.get('eta_mae', float('nan')):.1f} min")
        mc2.metric("ETA R²", f"{m.get('eta_r2', float('nan')):.3f}")
        mc3.metric("Delay ROC-AUC", f"{m.get('delay_roc_auc', float('nan')):.3f}")
