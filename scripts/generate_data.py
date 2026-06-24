"""Generate a synthetic order dataset and write it to data/processed.

Usage:
    python scripts/generate_data.py --n-orders 40000 --format parquet
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from delivery_delay.config import load_config
from delivery_delay.data.generator import add_targets, generate_orders


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic delivery orders")
    parser.add_argument("--n-orders", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--format", choices=["parquet", "csv"], default="csv")
    args = parser.parse_args()

    cfg = load_config()
    df = add_targets(generate_orders(cfg, n_orders=args.n_orders, seed=args.seed), cfg)

    out_dir = cfg.path("data_processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"orders.{args.format}"
    if args.format == "parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"Wrote {len(df):,} orders -> {out_path}")
    print(f"Delay rate: {df['is_delayed'].mean():.1%}")
    print(f"Median actual delivery time: {df['actual_minutes'].median():.1f} min")


if __name__ == "__main__":
    main()
