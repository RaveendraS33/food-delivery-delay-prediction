"""Train the ETA regressor and delay-probability classifier.

Two XGBoost models share the same feature matrix:

* ``eta_model``   -- regression on ``actual_minutes`` (the delivery ETA).
* ``delay_model`` -- binary classification of ``is_delayed`` (probability the
                     order misses its promised window by more than the
                     configured threshold).

Runs are tracked in MLflow (params, metrics, and the saved bundle as an
artifact) using a local file store by default, so nothing external is needed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from delivery_delay.config import Config, load_config
from delivery_delay.data.loader import load_canonical
from delivery_delay.features.build import build_xy, feature_columns
from delivery_delay.models.registry import ModelBundle, save_bundle

logger = logging.getLogger(__name__)


def _build_eta_model(cfg: Config) -> XGBRegressor:
    p = cfg.get("models.eta_regressor", {})
    return XGBRegressor(
        n_estimators=int(p.get("n_estimators", 400)),
        max_depth=int(p.get("max_depth", 7)),
        learning_rate=float(p.get("learning_rate", 0.05)),
        subsample=float(p.get("subsample", 0.9)),
        colsample_bytree=float(p.get("colsample_bytree", 0.9)),
        objective="reg:squarederror",
        random_state=cfg.seed,
        n_jobs=-1,
        tree_method="hist",
    )


def _build_delay_model(cfg: Config, scale_pos_weight: float) -> XGBClassifier:
    p = cfg.get("models.delay_classifier", {})
    return XGBClassifier(
        n_estimators=int(p.get("n_estimators", 400)),
        max_depth=int(p.get("max_depth", 6)),
        learning_rate=float(p.get("learning_rate", 0.05)),
        subsample=float(p.get("subsample", 0.9)),
        colsample_bytree=float(p.get("colsample_bytree", 0.9)),
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=cfg.seed,
        n_jobs=-1,
        tree_method="hist",
    )


def train_models(
    cfg: Config | None = None,
    source: str = "hybrid",
    n_orders: int | None = None,
    seed: int | None = None,
    log_mlflow: bool = True,
) -> ModelBundle:
    """Train both models end-to-end and persist the bundle. Returns the bundle."""
    cfg = cfg or load_config()
    seed = seed if seed is not None else cfg.seed

    logger.info("Loading data (source=%s)...", source)
    frame = load_canonical(cfg, source=source, n_orders=n_orders, seed=seed)
    X, y_eta, y_delay = build_xy(frame, cfg)
    logger.info("Training matrix: %d rows x %d features", X.shape[0], X.shape[1])

    X_tr, X_te, eta_tr, eta_te, del_tr, del_te = train_test_split(
        X, y_eta, y_delay, test_size=float(cfg.get("models.test_size", 0.2)), random_state=seed
    )

    # --- ETA regressor ---
    eta_model = _build_eta_model(cfg)
    eta_model.fit(X_tr, eta_tr)
    eta_pred = eta_model.predict(X_te)

    # --- Delay classifier (weight the minority "delayed" class) ---
    pos = max(int(del_tr.sum()), 1)
    neg = max(int((1 - del_tr).sum()), 1)
    spw = neg / pos
    delay_model = _build_delay_model(cfg, scale_pos_weight=spw)
    delay_model.fit(X_tr, del_tr)
    delay_prob = delay_model.predict_proba(X_te)[:, 1]

    metrics = {
        "eta_mae": float(mean_absolute_error(eta_te, eta_pred)),
        "eta_rmse": float(np.sqrt(mean_squared_error(eta_te, eta_pred))),
        "eta_r2": float(r2_score(eta_te, eta_pred)),
        "delay_rate": float(y_delay.mean()),
        "delay_roc_auc": float(roc_auc_score(del_te, delay_prob)),
        "delay_pr_auc": float(average_precision_score(del_te, delay_prob)),
        "delay_brier": float(brier_score_loss(del_te, delay_prob)),
    }
    logger.info(
        "ETA  MAE=%.2f min  RMSE=%.2f  R2=%.3f",
        metrics["eta_mae"],
        metrics["eta_rmse"],
        metrics["eta_r2"],
    )
    logger.info(
        "DELAY  ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.3f",
        metrics["delay_roc_auc"],
        metrics["delay_pr_auc"],
        metrics["delay_brier"],
    )

    bundle = ModelBundle(
        eta_model=eta_model,
        delay_model=delay_model,
        feature_columns=feature_columns(cfg),
        high_risk_threshold=float(cfg.get("models.delay_classifier.high_risk_threshold", 0.5)),
        delay_threshold_minutes=float(cfg.get("generator.delay_threshold_minutes", 10)),
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=source,
        n_train_rows=int(X_tr.shape[0]),
        metrics=metrics,
    )
    bundle_path = save_bundle(bundle, cfg.model_dir)
    logger.info("Saved model bundle -> %s", bundle_path)

    if log_mlflow:
        _log_to_mlflow(cfg, source, metrics, bundle_path)

    return bundle


def _log_to_mlflow(cfg: Config, source: str, metrics: dict, bundle_path) -> None:
    """Best-effort MLflow logging; never let tracking break a training run."""
    try:
        import os

        import mlflow

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "delivery-delay"))
        with mlflow.start_run():
            mlflow.log_param("source", source)
            mlflow.log_param("seed", cfg.seed)
            mlflow.log_params(
                {f"eta_{k}": v for k, v in cfg.get("models.eta_regressor", {}).items()}
            )
            mlflow.log_params(
                {f"delay_{k}": v for k, v in cfg.get("models.delay_classifier", {}).items()}
            )
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(bundle_path), artifact_path="model")
    except Exception as exc:  # pragma: no cover - tracking is optional
        logger.warning("MLflow logging skipped: %s", exc)
