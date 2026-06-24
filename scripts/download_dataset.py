"""Helper for fetching the optional public training dataset.

The public path is optional — the project trains end-to-end on synthetic data
out of the box. If you want to blend in a real dataset, this script will use the
Kaggle CLI when it is configured; otherwise it prints manual instructions.

Usage:
    python scripts/download_dataset.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import _bootstrap  # noqa: F401
from delivery_delay.config import load_config

# A commonly used food-delivery-time dataset on Kaggle.
KAGGLE_DATASET = "gauravmalik26/food-delivery-dataset"


def main() -> None:
    cfg = load_config()
    raw_dir = cfg.path("data_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("kaggle"):
        print(f"Downloading {KAGGLE_DATASET} via Kaggle CLI -> {raw_dir}")
        try:
            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    KAGGLE_DATASET,
                    "-p",
                    str(raw_dir),
                    "--unzip",
                ],
                check=True,
            )
            print("Done. Rename the relevant CSV to one of:")
            print("  data/raw/orders.csv | zomato_delivery.csv | deliverytime.csv")
            return
        except subprocess.CalledProcessError as exc:
            print(f"Kaggle download failed ({exc}); falling back to manual instructions.\n")

    print(
        "\nManual setup (optional):\n"
        "  1. Download a food-delivery-time dataset from Kaggle, e.g.\n"
        f"       https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
        "  2. Place the CSV at one of:\n"
        f"       {raw_dir / 'orders.csv'}\n"
        f"       {raw_dir / 'zomato_delivery.csv'}\n"
        f"       {raw_dir / 'deliverytime.csv'}\n"
        "  3. Run: python scripts/train.py --source hybrid\n"
        "\nExpected columns (best-effort mapping): Restaurant_latitude/longitude,\n"
        "Delivery_location_latitude/longitude, Time_taken(min), Road_traffic_density,\n"
        "Weatherconditions, Type_of_vehicle, Order_Date, Time_Orderd, multiple_deliveries.\n"
        "Missing columns are filled with neutral defaults — see src/delivery_delay/data/loader.py.\n"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
