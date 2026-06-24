"""Evaluate models: baselines, cross-validation, explainability, and plots.

Writes docs/model_report.md and docs/img/*.png.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --source synthetic --n-orders 40000
"""

from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401
from delivery_delay.models.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate delivery delay models")
    parser.add_argument("--source", choices=["synthetic", "public", "hybrid"], default="hybrid")
    parser.add_argument("--n-orders", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = evaluate(
        source=args.source, n_orders=args.n_orders, seed=args.seed, cv_folds=args.cv_folds
    )

    print("\n=== ETA regression (test split) ===")
    for row in result["eta_table"]:
        print(f"  {row[0]:<20} MAE={row[1]:>6}  RMSE={row[2]:>6}  R2={row[3]:>6}")
    print("\n=== Delay classification (test split) ===")
    for row in result["delay_table"]:
        print(
            f"  {row[0]:<20} ROC-AUC={row[1]:>6}  PR-AUC={row[2]:>6}  Brier={row[3]:>6}  F1={row[4]:>6}"
        )
    cv = result["cv"]
    print("\n=== 5-fold cross-validation (XGBoost) ===")
    print(f"  ETA MAE       {cv['eta_mae'][0]:.2f} +/- {cv['eta_mae'][1]:.2f} min")
    print(f"  ETA R2        {cv['eta_r2'][0]:.3f} +/- {cv['eta_r2'][1]:.3f}")
    print(f"  Delay ROC-AUC {cv['delay_roc_auc'][0]:.3f} +/- {cv['delay_roc_auc'][1]:.3f}")
    cb = result["cost_best"]
    print("\n=== Cost-aware operating point ===")
    print(
        f"  min-cost threshold {cb['threshold']:.2f} (vs 0.50): "
        f"precision={cb['precision']:.2f} recall={cb['recall']:.2f} F1={cb['f1']:.2f}"
    )
    print(f"\nReport + figures -> {result['out_dir']}")


if __name__ == "__main__":
    main()
