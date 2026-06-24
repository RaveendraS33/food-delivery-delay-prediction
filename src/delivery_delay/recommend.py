"""Optimal-ordering recommendation engine.

Given a route (restaurant -> customer), scan the next few hours and score each
candidate order time with the trained models. Surface the current risk, the
lowest-risk slot in the window, and a plain-language recommendation -- the
decision-support layer behind the dashboard's "best time to order" feature.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from delivery_delay.config import Config, load_config
from delivery_delay.models.predict import DelayPredictor

logger = logging.getLogger(__name__)


def recommend_order_time(
    predictor: DelayPredictor,
    restaurant_lat: float,
    restaurant_lon: float,
    customer_lat: float,
    customer_lon: float,
    start_time: datetime | None = None,
    lookahead_hours: int | None = None,
    cfg: Config | None = None,
    **context,
) -> dict:
    """Return current vs best order-time, an hourly schedule, and advice."""
    cfg = cfg or load_config()
    start_time = (start_time or datetime.now()).replace(minute=0, second=0, microsecond=0)
    lookahead = int(lookahead_hours or cfg.get("recommend.lookahead_hours", 6))
    acceptable = float(cfg.get("recommend.acceptable_delay_prob", 0.25))

    schedule: list[dict] = []
    for offset in range(lookahead + 1):
        t = start_time + timedelta(hours=offset)
        pred = predictor.predict_context(
            restaurant_lat=restaurant_lat,
            restaurant_lon=restaurant_lon,
            customer_lat=customer_lat,
            customer_lon=customer_lon,
            order_time=t,
            **context,
        )
        schedule.append(
            {
                "offset_hours": offset,
                "time": t.isoformat(timespec="minutes"),
                "hour": t.hour,
                "predicted_eta_minutes": pred["predicted_eta_minutes"],
                "delay_probability": pred["delay_probability"],
                "risk_level": pred["risk_level"],
            }
        )

    current = schedule[0]
    best = min(schedule, key=lambda s: (s["delay_probability"], s["offset_hours"]))

    if current["delay_probability"] <= acceptable:
        advice = "Good time to order now — low predicted delay risk."
        recommend_now = True
    elif best["offset_hours"] == 0:
        advice = "Demand is elevated across the window; ordering now is still your best option."
        recommend_now = True
    else:
        drop = current["delay_probability"] - best["delay_probability"]
        advice = (
            f"Consider waiting until {best['time'][11:16]} — predicted delay risk drops "
            f"{drop * 100:.0f} points (from {current['delay_probability'] * 100:.0f}% to "
            f"{best['delay_probability'] * 100:.0f}%)."
        )
        recommend_now = False

    logger.info(
        "Recommendation over %dh: now=%.0f%% best=%.0f%% @ +%dh",
        lookahead,
        current["delay_probability"] * 100,
        best["delay_probability"] * 100,
        best["offset_hours"],
    )
    return {
        "recommend_now": recommend_now,
        "advice": advice,
        "acceptable_delay_prob": acceptable,
        "current": current,
        "best": best,
        "schedule": schedule,
    }
