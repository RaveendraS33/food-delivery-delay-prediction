# Data Dictionary

All data sources (synthetic generator, public Kaggle loader, hybrid) emit the
same **canonical order schema**. The feature pipeline
([`features/build.py`](../src/delivery_delay/features/build.py)) consumes this
schema, and the serving path assembles a single canonical row per request — so
training and inference share one contract.

## Canonical order schema

| Column | Type | Unit / domain | Description |
|---|---|---|---|
| `order_id` | str | — | Unique order identifier |
| `timestamp` | datetime | local | When the order was placed (drives temporal features) |
| `restaurant_id` | str | — | Restaurant identifier |
| `restaurant_lat` | float | degrees | Restaurant latitude |
| `restaurant_lon` | float | degrees | Restaurant longitude |
| `customer_lat` | float | degrees | Delivery latitude |
| `customer_lon` | float | degrees | Delivery longitude |
| `prep_time_minutes` | float | minutes | Restaurant food-prep estimate |
| `traffic_level` | categorical | `low\|medium\|high\|jam` | Road traffic density |
| `vehicle_type` | categorical | `bike\|scooter\|car` | Courier vehicle |
| `active_orders` | int | count | Concurrent orders (kitchen/area load) |
| `weather_temp_c` | float | °C | Temperature at order time |
| `weather_precip_mm` | float | mm | Precipitation (0 = dry) |
| `weather_wind_kmph` | float | km/h | Wind speed |
| `promised_minutes` | float | minutes | Platform's quoted ETA (known at order time) |
| `actual_minutes` | float | minutes | **Observed** delivery time — ETA regression target |

## Targets (derived)

| Column | Type | Definition |
|---|---|---|
| `delay_minutes` | float | `actual_minutes - promised_minutes` |
| `is_delayed` | int (0/1) | `1` if `delay_minutes > delay_threshold_minutes` (config; default 8) — classification target |

> **Leakage note.** `actual_minutes`, `delay_minutes`, and `is_delayed` are
> **targets only** and never enter the feature matrix. `promised_minutes` is the
> platform quote, available *before* delivery, so it is a valid feature.

## Engineered features (model inputs)

Produced by `build_features` → 31 numeric columns:

| Group | Features |
|---|---|
| Geo | `distance_km` (haversine restaurant→customer) |
| Order context | `prep_time_minutes`, `active_orders`, `promised_minutes` |
| Weather | `weather_temp_c`, `weather_precip_mm`, `weather_wind_kmph` |
| Temporal | `hour`, `day_of_week`, `is_weekend`, `is_lunch_peak`, `is_dinner_peak`, `is_peak`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` |
| Traffic (one-hot) | `traffic_low`, `traffic_medium`, `traffic_high`, `traffic_jam` |
| Vehicle (one-hot) | `vehicle_bike`, `vehicle_scooter`, `vehicle_car` |
| Meal window (one-hot) | `meal_breakfast`, `meal_lunch`, `meal_dinner`, `meal_late`, `meal_off_peak` |
| Interactions | `load_x_peak` (active_orders × is_peak), `dist_x_rain` (distance × precip) |

Categoricals are one-hot encoded against **fixed vocabularies**, so the column
set is identical for a 1-row request and the full training set. Missing numeric
values are coerced and filled with `0.0`; the public loader drops rows lacking a
target or coordinates.
