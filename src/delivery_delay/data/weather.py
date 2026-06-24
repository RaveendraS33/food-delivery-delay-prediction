"""Open-Meteo weather client.

Open-Meteo is free and needs no API key, which keeps the project at $0 to run.
We use it for *live* forecasts on the serving path (API + dashboard). For
bulk training data we synthesize weather offline (see ``generator.py``) so we
never hammer the API with tens of thousands of calls.

The client caches responses in-memory with a TTL and degrades gracefully to a
neutral fallback when the network is unavailable, so predictions never hard
fail just because the weather lookup did.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests

DEFAULT_BASE_URL = os.getenv("WEATHER_BASE_URL", "https://api.open-meteo.com/v1/forecast")
DEFAULT_TTL = int(os.getenv("WEATHER_CACHE_TTL_SECONDS", "900"))

# Hourly variables we request from Open-Meteo.
_HOURLY_VARS = "temperature_2m,precipitation,wind_speed_10m"

# Neutral, mild conditions used when the API is unreachable.
FALLBACK_WEATHER = {
    "weather_temp_c": 18.0,
    "weather_precip_mm": 0.0,
    "weather_wind_kmph": 10.0,
    "source": "fallback",
}


@dataclass
class _CacheEntry:
    expires_at: float
    payload: dict


class WeatherClient:
    """Tiny TTL-cached client around the Open-Meteo forecast endpoint."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        ttl_seconds: int = DEFAULT_TTL,
        timeout: float = 4.0,
    ):
        self.base_url = base_url
        self.ttl = ttl_seconds
        self.timeout = timeout
        self._cache: dict[tuple, _CacheEntry] = {}

    def _cache_key(self, lat: float, lon: float) -> tuple:
        # Round coordinates so nearby points share a cached forecast.
        return (round(lat, 2), round(lon, 2))

    def get_forecast(self, lat: float, lon: float) -> dict | None:
        """Return the raw hourly forecast for a location (cached), or ``None``."""
        key = self._cache_key(lat, lon)
        now = time.time()
        cached = self._cache.get(key)
        if cached and cached.expires_at > now:
            return cached.payload

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": _HOURLY_VARS,
            "forecast_days": 2,
            "timezone": "auto",
        }
        try:
            resp = requests.get(self.base_url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError):
            return None

        self._cache[key] = _CacheEntry(expires_at=now + self.ttl, payload=payload)
        return payload

    def weather_at(self, lat: float, lon: float, iso_hour: str | None = None) -> dict:
        """Weather features for a location at a given hour.

        ``iso_hour`` is an ``YYYY-MM-DDTHH:00`` string matching Open-Meteo's
        hourly index. If omitted, the first (current) hour is used. Always
        returns a dict with the three weather feature keys; falls back to mild
        conditions when the forecast is unavailable.
        """
        forecast = self.get_forecast(lat, lon)
        if not forecast or "hourly" not in forecast:
            return dict(FALLBACK_WEATHER)

        hourly = forecast["hourly"]
        times: list[str] = hourly.get("time", [])
        if not times:
            return dict(FALLBACK_WEATHER)

        idx = 0
        if iso_hour is not None:
            # Match on the "YYYY-MM-DDTHH" prefix (ignore minutes).
            prefix = iso_hour[:13]
            for i, t in enumerate(times):
                if t[:13] == prefix:
                    idx = i
                    break

        def _at(name: str, default: float) -> float:
            series = hourly.get(name) or []
            if idx < len(series) and series[idx] is not None:
                return float(series[idx])
            return default

        return {
            "weather_temp_c": _at("temperature_2m", 18.0),
            "weather_precip_mm": _at("precipitation", 0.0),
            "weather_wind_kmph": _at("wind_speed_10m", 10.0),
            "source": "open-meteo",
        }


# Module-level default client for convenience.
default_client = WeatherClient()
