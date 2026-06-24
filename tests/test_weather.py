from delivery_delay.data.weather import FALLBACK_WEATHER, WeatherClient


def test_fallback_when_unreachable():
    # Unroutable endpoint -> client must degrade to fallback conditions.
    client = WeatherClient(base_url="http://127.0.0.1:9/forecast", ttl_seconds=60, timeout=1.0)
    w = client.weather_at(42.36, -71.06)
    assert w["source"] == "fallback"
    for key in ("weather_temp_c", "weather_precip_mm", "weather_wind_kmph"):
        assert key in w


def test_fallback_values_are_neutral():
    client = WeatherClient(base_url="http://127.0.0.1:9/forecast", timeout=1.0)
    w = client.weather_at(0.0, 0.0)
    assert w["weather_precip_mm"] == FALLBACK_WEATHER["weather_precip_mm"]
