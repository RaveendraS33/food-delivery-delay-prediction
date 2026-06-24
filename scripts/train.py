"""Train the ETA + delay models and save the bundle.

Usage:
    python scripts/train.py                      # hybrid source (public if present, else synthetic)
    python scripts/train.py --source synthetic --n-orders 20000
    python scripts/train.py --no-mlflow
"""

from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401  (sys.path side effect)
from delivery_delay.models.train import train_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train delivery delay models")
    parser.add_argument("--source", choices=["synthetic", "public", "hybrid"], default="hybrid")
    parser.add_argument("--n-orders", type=int, default=None, help="synthetic order count")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    bundle = train_models(
        source=args.source,
        n_orders=args.n_orders,
        seed=args.seed,
        log_mlflow=not args.no_mlflow,
    )

    print("\n=== Training complete ===")
    print(f"source        : {bundle.source}")
    print(f"train rows    : {bundle.n_train_rows}")
    print(f"features      : {len(bundle.feature_columns)}")
    for k, v in bundle.metrics.items():
        print(f"{k:<16}: {v:.4f}")


if __name__ == "__main__":
    main()
