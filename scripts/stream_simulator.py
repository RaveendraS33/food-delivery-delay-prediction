"""Simulate a live stream of order events hitting the prediction API.

Demonstrates the real-time serving path: generates plausible orders and POSTs
them to ``/predict`` at a configurable rate, printing the ETA + delay risk the
service returns. Run the API first (``make api`` or docker-compose).

Usage:
    python scripts/stream_simulator.py --rate 1.5 --n 50
    python scripts/stream_simulator.py --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import requests

import _bootstrap  # noqa: F401
from delivery_delay.config import load_config


def _random_order(cfg, rng) -> dict:
    lat_min = cfg.get("city.lat_min", 42.31)
    lat_max = cfg.get("city.lat_max", 42.39)
    lon_min = cfg.get("city.lon_min", -71.12)
    lon_max = cfg.get("city.lon_max", -71.03)
    r_lat = rng.uniform(lat_min, lat_max)
    r_lon = rng.uniform(lon_min, lon_max)
    return {
        "restaurant_lat": round(r_lat, 5),
        "restaurant_lon": round(r_lon, 5),
        "customer_lat": round(float(np.clip(r_lat + rng.normal(0, 0.012), lat_min, lat_max)), 5),
        "customer_lon": round(float(np.clip(r_lon + rng.normal(0, 0.015), lon_min, lon_max)), 5),
        "vehicle_type": rng.choice(["bike", "scooter", "car"]),
        "active_orders": int(rng.integers(1, 18)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream synthetic orders to the API")
    parser.add_argument("--url", default=os.getenv("DELIVERY_API_URL", "http://localhost:8000"))
    parser.add_argument("--rate", type=float, default=1.0, help="orders per second")
    parser.add_argument("--n", type=int, default=30, help="number of orders to send")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    rng = np.random.default_rng(args.seed if args.seed is not None else cfg.seed)
    interval = 1.0 / max(args.rate, 0.01)
    endpoint = args.url.rstrip("/") + "/predict"

    print(f"Streaming {args.n} orders to {endpoint} at {args.rate}/s\n")
    sent = 0
    for _ in range(args.n):
        order = _random_order(cfg, rng)
        try:
            resp = requests.post(endpoint, json=order, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            print(
                f"[{sent:>3}] dist={data['distance_km']:>4.1f}km  "
                f"ETA={data['predicted_eta_minutes']:>5.1f}min  "
                f"delay_risk={data['delay_probability'] * 100:>5.1f}%  "
                f"({data['risk_level']})"
            )
        except requests.RequestException as exc:
            print(f"[{sent:>3}] request failed: {exc}")
        sent += 1
        time.sleep(interval)


if __name__ == "__main__":
    main()
