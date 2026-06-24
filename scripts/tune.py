"""Tune the delay classifier's hyperparameters (randomized search).

Usage:
    python scripts/tune.py                 # 12 candidates, 3-fold, 15k rows
    python scripts/tune.py --n-iter 30 --cv 5 --n-orders 30000
"""

from __future__ import annotations

import argparse
import json
import logging

import _bootstrap  # noqa: F401
from delivery_delay.models.tune import tune_delay_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomized hyperparameter search")
    parser.add_argument("--source", choices=["synthetic", "public", "hybrid"], default="synthetic")
    parser.add_argument("--n-orders", type=int, default=15000)
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    best = tune_delay_model(
        source=args.source, n_orders=args.n_orders, n_iter=args.n_iter, cv=args.cv, seed=args.seed
    )

    print("\n=== Best (ROC-AUC = {:.4f}) ===".format(best["best_score_roc_auc"]))
    print(json.dumps(best["best_params"], indent=2))


if __name__ == "__main__":
    main()
