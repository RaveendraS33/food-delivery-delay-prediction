# Data

The project runs end-to-end on **synthetic data** out of the box — you do not
need to download anything. The optional public-dataset path lets you blend a
real Kaggle food-delivery dataset into training (the `hybrid` source).

## Layout

```
data/
├── raw/          # drop the public CSV here (git-ignored)
└── processed/    # generated synthetic datasets land here (git-ignored)
```

## Optional: add a public dataset

1. Download a food-delivery-time dataset from Kaggle, e.g.
   [`gauravmalik26/food-delivery-dataset`](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset)
   (or any similar Zomato/Swiggy delivery dataset).
2. Place the CSV at one of these paths (the loader checks them in order):
   - `data/raw/orders.csv`
   - `data/raw/zomato_delivery.csv`
   - `data/raw/deliverytime.csv`
3. Train with the hybrid source:
   ```bash
   python scripts/train.py --source hybrid
   ```

`scripts/download_dataset.py` will do steps 1–2 automatically if the Kaggle CLI
is configured.

## Column mapping (best effort)

The loader normalises column names and maps the common Zomato/Swiggy schema into
the project's canonical schema. Recognised columns include:

| Public column | Canonical field |
|---|---|
| `Restaurant_latitude` / `Restaurant_longitude` | `restaurant_lat` / `restaurant_lon` |
| `Delivery_location_latitude` / `Delivery_location_longitude` | `customer_lat` / `customer_lon` |
| `Time_taken(min)` | `actual_minutes` (target) |
| `Road_traffic_density` | `traffic_level` |
| `Weatherconditions` | `weather_precip_mm` (proxy) |
| `Type_of_vehicle` | `vehicle_type` |
| `Order_Date` + `Time_Orderd` | `timestamp` |
| `multiple_deliveries` | `active_orders` (proxy) |

Anything missing is filled with neutral defaults, and `promised_minutes` is
derived from a distance + prep prior so the delay label is consistent across
sources. See [`src/delivery_delay/data/loader.py`](../src/delivery_delay/data/loader.py).
