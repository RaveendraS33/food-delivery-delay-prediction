"""Shared fixtures.

A single small model is trained once per test session into a temporary
MODEL_DIR so the inference and API tests have something real to score against,
without depending on a model committed to the repo.
"""

from __future__ import annotations

import os

# Point the weather client at an unreachable address so tests never hit the
# network: the client fails fast and returns its mild fallback conditions.
# Must be set before anything imports delivery_delay.data.weather.
os.environ.setdefault("WEATHER_BASE_URL", "http://127.0.0.1:9/forecast")

import pytest  # noqa: E402

from delivery_delay.config import load_config  # noqa: E402
from delivery_delay.models.train import train_models  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("models")
    os.environ["MODEL_DIR"] = str(model_dir)
    # Small synthetic run keeps the suite fast but exercises the full pipeline.
    train_models(source="synthetic", n_orders=4000, seed=7, log_mlflow=False)
    yield model_dir
    os.environ.pop("MODEL_DIR", None)


@pytest.fixture(scope="session")
def predictor(trained_model_dir):
    from delivery_delay.models.predict import DelayPredictor

    return DelayPredictor(model_dir=trained_model_dir)
