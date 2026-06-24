# 🛵 Food Delivery Delay Prediction System

End-to-end machine-learning system that predicts **delivery ETA** and the
**probability an order arrives late**, then recommends the **best time to order**
— combining live weather, geospatial distance, traffic, kitchen load, and
time-of-day features into a production-style decision-support service.

[![CI](https://github.com/RaveendraS33/food-delivery-delay-prediction/actions/workflows/ci.yml/badge.svg)](../../actions)
&nbsp;·&nbsp; Python 3.10–3.12 &nbsp;·&nbsp; FastAPI · XGBoost · MLflow · Streamlit · Docker &nbsp;·&nbsp; runs at **$0**

> Two models share one feature pipeline: an **ETA regressor** (`actual_minutes`)
> and a **delay-probability classifier** (`is_delayed`). The same
> `build_features` code runs at training and at serving time, so there is no
> train/serve skew.

---

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │                Data sources                  │
   Open-Meteo (live) ───▶│  synthetic generator  +  public Kaggle CSV   │
   weather, no API key   │            (hybrid loader)                    │
                         └───────────────────────┬─────────────────────┘
                                                 │  canonical schema
                                                 ▼
                                   ┌──────────────────────────┐
                                   │   Feature engineering     │
                                   │  geo · temporal · weather │
                                   │  traffic · kitchen load   │
                                   └─────────────┬─────────────┘
                                                 ▼
                         ┌───────────────────────────────────────────┐
                         │     Training  (XGBoost, tracked in MLflow)  │
                         │   ETA regressor  +  delay classifier        │
                         └───────────────────────┬─────────────────────┘
                                                 │  model_bundle.joblib
                            ┌────────────────────┴───────────────────┐
                            ▼                                         ▼
              ┌──────────────────────────┐              ┌──────────────────────────┐
              │     FastAPI service       │◀── /metrics ─┤  Prometheus + Grafana     │
              │  /predict  /recommend     │              │      (optional)           │
              └─────────────┬─────────────┘              └──────────────────────────┘
                            ▼
              ┌──────────────────────────┐
              │   Streamlit dashboard     │   ETA · delay-risk gauge ·
              │  decision-support UI      │   "best time to order"
              └──────────────────────────┘
```

Everything runs locally via Docker Compose and is structured to lift into the
cloud (see [Cloud-ready](#cloud-ready)).

---

## Results

Trained on 40,000 synthetic orders (hybrid source), 80/20 split:

| Model | Metric | Score |
|---|---|---|
| **ETA regressor** | MAE | **2.1 min** |
| | RMSE | 2.7 min |
| | R² | **0.91** |
| **Delay classifier** | ROC-AUC | **0.95** |
| | PR-AUC | 0.89 |
| | Brier score | 0.085 |
| Dataset | base delay rate | 27% |

The models recover the intended structure — e.g. a long route at the **19:00
dinner peak with a busy kitchen** scores **~92% delay risk**, while the **same
route at 15:00 off-peak** scores **~3%**, and the recommender suggests shifting
the order to a lower-risk slot.

---

## Quickstart

### Option A — local (Python)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python scripts/train.py            # train ETA + delay models -> models/
make api                           # FastAPI on http://localhost:8000/docs
make dashboard                     # Streamlit on http://localhost:8501
```

Send a prediction:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "restaurant_lat": 42.36, "restaurant_lon": -71.06,
  "customer_lat": 42.39,  "customer_lon": -71.11,
  "order_time": "2025-06-07T19:00:00", "active_orders": 16
}'
```

Stream a live feed of orders at the API:

```bash
python scripts/stream_simulator.py --rate 2 --n 40
```

### Option B — Docker Compose

```bash
docker compose run --rm trainer            # train into the shared ./models volume
docker compose up api dashboard            # API :8000 + dashboard :8501
docker compose --profile monitoring up     # + Prometheus :9090 and Grafana :3000
```

---

## How it works

**Data (hybrid).** A [synthetic generator](src/delivery_delay/data/generator.py)
produces realistic orders with baked-in cause→effect (distance, peak hours,
rain, traffic, kitchen load all push delivery time up). An optional
[public-dataset loader](src/delivery_delay/data/loader.py) maps a real Kaggle
delivery dataset into the same canonical schema (see [data/README.md](data/README.md)).
Live forecasts come from [Open-Meteo](src/delivery_delay/data/weather.py) — free,
no API key, with an offline fallback so predictions never hard-fail.

**Features.** [`build_features`](src/delivery_delay/features/build.py) assembles
geo distance (haversine), cyclical time encodings, peak/meal-window flags,
weather, and one-hot traffic/vehicle/meal categories against **fixed
vocabularies** — so a single request produces exactly the same columns as the
training set.

**Models.** Two XGBoost models ([train.py](src/delivery_delay/models/train.py))
share the feature matrix; the classifier weights the minority "delayed" class.
Runs are tracked in **MLflow** (params, metrics, model artifact).

**Serving.** A [FastAPI service](src/delivery_delay/api/main.py) exposes
`/predict`, `/recommend`, `/model/info`, and Prometheus `/metrics`. The
[recommendation engine](src/delivery_delay/recommend.py) scans the next few
hours and surfaces the lowest-risk time to order.

**Dashboard.** A [Streamlit app](dashboard/app.py) renders ETA, a delay-risk
gauge, the route map, and the "best time to order" schedule. It talks to the API
and transparently falls back to running the model in-process.

---

## Project structure

```
src/delivery_delay/
├── config.py              # YAML + env config
├── data/                  # generator, weather client, public loader
├── features/              # geo, temporal, build (train/serve parity)
├── models/                # train, registry, predict
├── recommend.py           # optimal-ordering engine
└── api/                   # FastAPI app, schemas, Prometheus metrics
dashboard/app.py           # Streamlit decision-support UI
scripts/                   # train · generate_data · stream_simulator · download_dataset
monitoring/                # Prometheus + Grafana provisioning
tests/                     # pytest: geo, temporal, features, generator, predict, api, weather
```

## Tech stack

`Python` · `scikit-learn` · `XGBoost` · `MLflow` · `FastAPI` · `Pydantic` ·
`Streamlit` · `Plotly` · `Prometheus` · `Grafana` · `Docker Compose` ·
`pytest` · `ruff`/`black` · `GitHub Actions`

## Cloud-ready

The local stack maps cleanly onto AWS: containerised API/dashboard → **ECS/Fargate**
(or the API on **Lambda + API Gateway**), the model bundle in **S3** with the
MLflow registry, batch training on **SageMaker/Batch**, and metrics shipped to
**CloudWatch / Managed Prometheus + Grafana**. Config and artifact paths are
already environment-driven for exactly this.

## Testing

```bash
make test     # pytest (hermetic: weather is stubbed, a small model trains in-fixture)
make lint     # ruff
```

CI runs lint + format-check + the full suite on Python 3.10/3.11/3.12.

## License

MIT
