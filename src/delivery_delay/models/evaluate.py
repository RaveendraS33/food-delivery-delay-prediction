"""Model evaluation, baselines, cross-validation and explainability.

Produces the artifacts a reviewer expects beyond a single accuracy number:

* a **baseline comparison** (naive + linear models vs XGBoost) so the gains are
  attributable, not assumed;
* **k-fold cross-validation** (mean +/- std) so the headline metrics aren't a
  lucky split;
* **feature importance** and **SHAP** explanations;
* diagnostic plots: predicted-vs-actual, residuals, ROC, calibration,
  confusion matrix, and delay-rate-by-hour.

Outputs a Markdown report (``docs/model_report.md``) and PNGs under
``docs/img/`` that the README embeds. Run via ``python scripts/evaluate.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibrationDisplay  # noqa: E402
from sklearn.dummy import DummyClassifier, DummyRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from delivery_delay.config import PROJECT_ROOT, Config, load_config  # noqa: E402
from delivery_delay.data.loader import load_canonical  # noqa: E402
from delivery_delay.features.build import build_xy  # noqa: E402
from delivery_delay.models.train import _build_delay_model, _build_eta_model  # noqa: E402

logger = logging.getLogger(__name__)


def _savefig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _threshold_sweep(y_true, prob, cost_fn: float, cost_fp: float, img_dir: Path) -> dict:
    """Sweep decision thresholds; pick the one minimising expected cost/order.

    A missed delay (false negative -> SLA breach / refund / lost trust) is more
    expensive than a false alarm (false positive -> a proactive nudge or padded
    ETA), so we weight ``cost_fn`` > ``cost_fp`` and choose the operating point
    accordingly instead of defaulting to 0.5.
    """
    y = np.asarray(y_true)
    ts = np.linspace(0.05, 0.95, 19)
    prec_l, rec_l, f1_l, cost_l = [], [], [], []
    best = None
    for t in ts:
        pred = (prob >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        cost = (fn * cost_fn + fp * cost_fp) / len(y)
        prec_l.append(prec)
        rec_l.append(rec)
        f1_l.append(f1)
        cost_l.append(cost)
        if best is None or cost < best["cost"]:
            best = {
                "threshold": float(t),
                "cost": float(cost),
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(ts, prec_l, label="precision")
    ax[0].plot(ts, rec_l, label="recall")
    ax[0].plot(ts, f1_l, label="F1")
    ax[0].axvline(0.5, color="gray", ls=":", label="default 0.5")
    ax[0].set_xlabel("decision threshold")
    ax[0].legend()
    ax[0].set_title("Precision / recall / F1")
    ax[1].plot(ts, cost_l, color="#ef4444")
    ax[1].axvline(
        best["threshold"], color="green", ls="--", label=f"min-cost @ {best['threshold']:.2f}"
    )
    ax[1].set_xlabel("decision threshold")
    ax[1].set_ylabel(f"expected cost/order (FN={cost_fn:g}, FP={cost_fp:g})")
    ax[1].legend()
    ax[1].set_title("Expected cost vs threshold")
    _savefig(fig, img_dir / "threshold_sweep.png")
    return best


def evaluate(
    cfg: Config | None = None,
    source: str = "hybrid",
    n_orders: int | None = None,
    seed: int | None = None,
    out_dir: str | Path | None = None,
    cv_folds: int = 5,
    shap_sample: int = 2000,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
) -> dict:
    """Run the full evaluation and write report + figures. Returns a summary dict."""
    cfg = cfg or load_config()
    seed = seed if seed is not None else cfg.seed
    out_dir = Path(out_dir) if out_dir else PROJECT_ROOT / "docs"
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data (source=%s)...", source)
    frame = load_canonical(cfg, source=source, n_orders=n_orders, seed=seed)
    X, y_eta, y_delay = build_xy(frame, cfg)

    X_tr, X_te, eta_tr, eta_te, del_tr, del_te = train_test_split(
        X,
        y_eta,
        y_delay,
        test_size=float(cfg.get("models.test_size", 0.2)),
        random_state=seed,
        stratify=y_delay,
    )
    spw = max(int((1 - del_tr).sum()), 1) / max(int(del_tr.sum()), 1)

    # ---------------- ETA: baselines vs XGBoost ----------------
    eta_models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "XGBoost": _build_eta_model(cfg),
    }
    eta_rows, eta_pred_xgb = [], None
    for name, model in eta_models.items():
        model.fit(X_tr, eta_tr)
        pred = model.predict(X_te)
        eta_rows.append(
            [
                name,
                f"{mean_absolute_error(eta_te, pred):.2f}",
                f"{np.sqrt(mean_squared_error(eta_te, pred)):.2f}",
                f"{r2_score(eta_te, pred):.3f}",
            ]
        )
        if name == "XGBoost":
            eta_pred_xgb = pred

    # ---------------- Delay: baselines vs XGBoost ----------------
    delay_models = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "XGBoost": _build_delay_model(cfg, scale_pos_weight=spw),
    }
    delay_rows, delay_prob_xgb = [], None
    roc_fig, roc_ax = plt.subplots(figsize=(5.5, 4.5))
    for name, model in delay_models.items():
        model.fit(X_tr, del_tr)
        prob = model.predict_proba(X_te)[:, 1]
        preds = (prob >= 0.5).astype(int)
        delay_rows.append(
            [
                name,
                f"{roc_auc_score(del_te, prob):.3f}",
                f"{average_precision_score(del_te, prob):.3f}",
                f"{brier_score_loss(del_te, prob):.3f}",
                f"{f1_score(del_te, preds, zero_division=0):.3f}",
            ]
        )
        RocCurveDisplay.from_predictions(del_te, prob, name=name, ax=roc_ax)
        if name == "XGBoost":
            delay_prob_xgb = prob
            delay_best = model
    roc_ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    roc_ax.set_title("Delay classifier — ROC")
    _savefig(roc_fig, img_dir / "delay_roc.png")

    # Cost-aware operating point (a missed delay costs more than a false alarm).
    cost_best = _threshold_sweep(del_te, delay_prob_xgb, cost_fn, cost_fp, img_dir)
    logger.info(
        "Min-cost threshold=%.2f (vs default 0.50): precision=%.2f recall=%.2f",
        cost_best["threshold"],
        cost_best["precision"],
        cost_best["recall"],
    )

    # ---------------- Cross-validation (the chosen XGBoost models) ----------------
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_mae = -cross_val_score(
        _build_eta_model(cfg), X, y_eta, cv=cv_folds, scoring="neg_mean_absolute_error"
    )
    cv_r2 = cross_val_score(_build_eta_model(cfg), X, y_eta, cv=cv_folds, scoring="r2")
    cv_auc = cross_val_score(_build_delay_model(cfg, spw), X, y_delay, cv=skf, scoring="roc_auc")
    cv = {
        "eta_mae": (cv_mae.mean(), cv_mae.std()),
        "eta_r2": (cv_r2.mean(), cv_r2.std()),
        "delay_roc_auc": (cv_auc.mean(), cv_auc.std()),
    }

    # ---------------- Diagnostic plots ----------------
    # Predicted vs actual ETA
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(eta_te, eta_pred_xgb, s=4, alpha=0.2)
    lims = [min(eta_te.min(), eta_pred_xgb.min()), max(eta_te.max(), eta_pred_xgb.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlabel("Actual ETA (min)")
    ax.set_ylabel("Predicted ETA (min)")
    ax.set_title("ETA: predicted vs actual")
    _savefig(fig, img_dir / "eta_pred_vs_actual.png")

    # Residual histogram
    resid = eta_pred_xgb - eta_te.to_numpy()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(resid, bins=60, color="#2563eb", alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Residual (pred - actual), min")
    ax.set_ylabel("Count")
    ax.set_title(f"ETA residuals (MAE={mean_absolute_error(eta_te, eta_pred_xgb):.2f} min)")
    _savefig(fig, img_dir / "eta_residuals.png")

    # Confusion matrix (delay @ 0.5)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        del_te,
        (delay_prob_xgb >= 0.5).astype(int),
        display_labels=["on-time", "delayed"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Delay confusion matrix (@0.5)")
    _savefig(fig, img_dir / "delay_confusion_matrix.png")

    # Calibration
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    CalibrationDisplay.from_predictions(del_te, delay_prob_xgb, n_bins=10, ax=ax)
    ax.set_title("Delay classifier — calibration")
    _savefig(fig, img_dir / "delay_calibration.png")

    # Feature importance (delay model, gain)
    importances = pd.Series(delay_best.feature_importances_, index=X.columns).sort_values()[-15:]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    importances.plot.barh(ax=ax, color="#0ea5e9")
    ax.set_title("Top features — delay classifier (XGBoost gain)")
    ax.set_xlabel("Importance")
    _savefig(fig, img_dir / "feature_importance.png")

    # Delay rate by hour (EDA-style operational view)
    by_hour = (
        frame.assign(hour=pd.to_datetime(frame["timestamp"]).dt.hour)
        .groupby("hour")["is_delayed"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_hour.index, by_hour.values * 100, color="#f59e0b")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Delay rate (%)")
    ax.set_title("Delay rate by order hour (peaks drive risk)")
    _savefig(fig, img_dir / "delay_rate_by_hour.png")

    # SHAP summary (best-effort; skip cleanly if shap unavailable)
    shap_ok = _shap_summary(delay_best, X_te, img_dir / "shap_summary.png", shap_sample, seed)

    # ---------------- Report ----------------
    report = _build_report(eta_rows, delay_rows, cv, frame, shap_ok, cost_best, cost_fn, cost_fp)
    (out_dir / "model_report.md").write_text(report, encoding="utf-8")
    logger.info("Wrote report -> %s", out_dir / "model_report.md")

    return {
        "eta_table": eta_rows,
        "delay_table": delay_rows,
        "cv": cv,
        "cost_best": cost_best,
        "shap": shap_ok,
        "out_dir": str(out_dir),
    }


def _shap_summary(model, X_te, path: Path, sample: int, seed: int) -> bool:
    try:
        import shap

        Xs = X_te.sample(min(sample, len(X_te)), random_state=seed)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(Xs)
        plt.figure()
        shap.summary_plot(values, Xs, show=False, plot_size=(7, 5))
        plt.title("SHAP — delay classifier")
        plt.savefig(path, dpi=110, bbox_inches="tight")
        plt.close()
        return True
    except Exception as exc:  # pragma: no cover - shap optional
        logger.warning("SHAP summary skipped: %s", exc)
        return False


def _build_report(eta_rows, delay_rows, cv, frame, shap_ok, cost_best, cost_fn, cost_fp) -> str:
    parts = [
        "# Model Evaluation Report",
        "",
        "_Auto-generated by `python scripts/evaluate.py`._",
        "",
        f"Dataset: **{len(frame):,}** orders · base delay rate **{frame['is_delayed'].mean():.1%}** "
        f"· median actual delivery **{frame['actual_minutes'].median():.1f} min**.",
        "",
        "## ETA regression — baseline comparison",
        _md_table(["Model", "MAE (min)", "RMSE (min)", "R²"], eta_rows),
        "",
        "## Delay classification — baseline comparison",
        _md_table(["Model", "ROC-AUC", "PR-AUC", "Brier", "F1@0.5"], delay_rows),
        "",
        "## Cross-validation (XGBoost, k=5)",
        _md_table(
            ["Metric", "Mean", "Std"],
            [
                ["ETA MAE (min)", f"{cv['eta_mae'][0]:.2f}", f"{cv['eta_mae'][1]:.2f}"],
                ["ETA R²", f"{cv['eta_r2'][0]:.3f}", f"{cv['eta_r2'][1]:.3f}"],
                ["Delay ROC-AUC", f"{cv['delay_roc_auc'][0]:.3f}", f"{cv['delay_roc_auc'][1]:.3f}"],
            ],
        ),
        "",
        "## Business framing & operating point",
        "",
        f"Errors are not symmetric: a **missed delay** (false negative → SLA breach, refund, "
        f"lost trust) is treated as **{cost_fn:g}×** the cost of a **false alarm** (false "
        f"positive → a proactive nudge or padded ETA). Sweeping the decision threshold to "
        f"minimise expected cost/order yields an operating point of **{cost_best['threshold']:.2f}** "
        f"(vs the naive 0.50) — precision {cost_best['precision']:.2f}, recall "
        f"{cost_best['recall']:.2f}, F1 {cost_best['f1']:.2f}. Tune `cost_fn`/`cost_fp` in "
        "`scripts/evaluate.py` to your real economics.",
        "",
        "![](img/threshold_sweep.png)",
        "",
        "## Diagnostics",
        "",
        "| | |",
        "|---|---|",
        "| ![](img/eta_pred_vs_actual.png) | ![](img/eta_residuals.png) |",
        "| ![](img/delay_roc.png) | ![](img/delay_calibration.png) |",
        "| ![](img/delay_confusion_matrix.png) | ![](img/delay_rate_by_hour.png) |",
        "",
        "## Explainability",
        "",
        "![](img/feature_importance.png)",
        "",
    ]
    if shap_ok:
        parts += ["![](img/shap_summary.png)", ""]
    return "\n".join(parts)
