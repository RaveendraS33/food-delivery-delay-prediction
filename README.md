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

## Results & validation

Evaluated on 40,000 orders (synthetic source), 80/20 split **stratified on the
delay label**. Naive + linear baselines are included so the gains are
*attributable, not assumed*. Full report and all diagnostic plots:
**[docs/model_report.md](docs/model_report.md)** (regenerate with `python scripts/evaluate.py`).

**ETA regression**

| Model | MAE (min) | RMSE (min) | R² |
|---|---|---|---|
| Mean baseline | 7.07 | 8.98 | 0.00 |
| Linear regression | 2.23 | 2.81 | 0.902 |
| **XGBoost** | **2.14** | **2.70** | **0.910** |

**Delay classification**

| Model | ROC-AUC | PR-AUC | Brier | F1@0.5 |
|---|---|---|---|---|
| Majority baseline | 0.500 | 0.276 | 0.276 | 0.00 |
| Logistic regression | 0.956 | 0.906 | 0.084 | 0.803 |
| **XGBoost** | **0.954** | 0.898 | **0.082** | **0.808** |

**5-fold cross-validation (XGBoost):** ETA MAE **2.14 ± 0.01** min · R² **0.910 ± 0.002** ·
delay ROC-AUC **0.951 ± 0.001** — the tiny variance rules out a lucky split.

> A linear/logistic baseline is already strong here (the relationships are fairly
> smooth), so XGBoost is chosen for its better calibration, F1, and capture of
> interactions (peak × load, distance × rain) — the comparison keeps that choice honest.

<table>
  <tr>
    <td><img src="docs/img/feature_importance.png" width="390" alt="feature importance"></td>
    <td><img src="docs/img/shap_summary.png" width="390" alt="SHAP summary"></td>
  </tr>
  <tr>
    <td><img src="docs/img/delay_calibration.png" width="390" alt="calibration curve"></td>
    <td><img src="docs/img/delay_rate_by_hour.png" width="390" alt="delay rate by hour"></td>
  </tr>
</table>

**Sanity check.** A long route at the **19:00 dinner peak with a busy kitchen**
scores **~92% delay risk**; the **same route at 15:00 off-peak** scores **~3%**,
and the recommender suggests the lower-risk slot.

### Operating point (cost-aware)

Errors aren't symmetric — a **missed delay** (SLA breach / refund / lost trust)
is costlier than a **false alarm** (a proactive nudge or padded ETA). Weighting a
false negative at **5×** a false positive and sweeping the threshold to minimise
expected cost/order moves the operating point to **0.25** (vs the naive 0.50):
**recall 0.94**, precision 0.63. The weights live in `scripts/evaluate.py` — set
them to your real economics. Hyperparameters are confirmed by a randomized search
([docs/tuning_report.md](docs/tuning_report.md)); the config defaults are already near-optimal.

![](docs/img/threshold_sweep.png)

### Validation & leakage controls
- 80/20 split **stratified** on `is_delayed`; headline metrics confirmed by 5-fold CV.
- **Train/serve parity** — one `build_features` path, categoricals one-hot encoded
  against fixed vocabularies, so a single request yields exactly the training columns.
- **No leakage** — `actual_minutes` / `delay_minutes` / `is_delayed` are targets only;
  `promised_minutes` (the platform's quote) is known at order time and is a valid feature.

📓 **Walkthrough:** [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) — EDA narrative,
model justification, and error analysis. &nbsp;📖 **Schema:** [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md).

---

## Quickstart

### Option A — local (Python)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                 # or: conda env create -f environment.yml

python scripts/train.py            # train ETA + delay models -> models/
python scripts/evaluate.py         # baselines, CV, SHAP, cost sweep -> docs/ (optional)
python scripts/tune.py             # randomized hyperparameter search (optional)
make api                           # FastAPI on http://localhost:8000/docs
make dashboard                     # Streamlit on http://localhost:8501
```

A narrated walkthrough lives in [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb).

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
Runs are tracked in **MLflow**, and each trained bundle is **versioned** (a
timestamped copy is archived alongside the latest).

**Evaluation.** [evaluate.py](src/delivery_delay/models/evaluate.py) benchmarks
both models against naive + linear baselines, cross-validates, computes feature
importance + **SHAP**, and runs a **cost-aware threshold sweep** — writing
[docs/model_report.md](docs/model_report.md) and the plots above.
[tune.py](src/delivery_delay/models/tune.py) does a randomized hyperparameter search.

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
├── data/                  # generator, weather client, public loader + schema validation
├── features/              # geo, temporal, build (train/serve parity)
├── models/                # train, evaluate, tune, registry (versioned), predict
├── recommend.py           # optimal-ordering engine
└── api/                   # FastAPI app, schemas, Prometheus metrics
dashboard/app.py           # Streamlit decision-support UI
scripts/                   # train · evaluate · tune · generate_data · stream_simulator
notebooks/01_eda.ipynb     # narrated EDA, model justification, error analysis
docs/                      # DATA_DICTIONARY.md · model_report.md · tuning_report.md · img/
monitoring/                # Prometheus + Grafana provisioning
tests/                     # pytest: geo, temporal, features, generator, predict, api, weather
```

## Tech stack

`Python` · `scikit-learn` · `XGBoost` · `SHAP` · `MLflow` · `FastAPI` · `Pydantic` ·
`Streamlit` · `Plotly` · `matplotlib`/`seaborn` · `Jupyter` · `Prometheus` · `Grafana` ·
`Docker Compose` · `pytest` · `ruff`/`black` · `GitHub Actions`

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
