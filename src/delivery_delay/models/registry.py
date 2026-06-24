"""Model persistence.

The trained artifact is a single ``ModelBundle`` (both models + the feature
column contract + metadata) saved with joblib, alongside a human-readable
``metadata.json``. Keeping the feature column list inside the bundle is what
lets the serving path reindex any incoming row to the exact training columns.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib

BUNDLE_FILENAME = "model_bundle.joblib"
METADATA_FILENAME = "metadata.json"


@dataclass
class ModelBundle:
    eta_model: Any
    delay_model: Any
    feature_columns: list[str]
    high_risk_threshold: float = 0.5
    delay_threshold_minutes: float = 10.0
    trained_at: str = ""
    source: str = "synthetic"
    n_train_rows: int = 0
    metrics: dict[str, float] = field(default_factory=dict)

    def metadata(self) -> dict:
        """JSON-serialisable subset (everything except the model objects)."""
        d = asdict(self)
        d.pop("eta_model", None)
        d.pop("delay_model", None)
        return d


def save_bundle(bundle: ModelBundle, model_dir: str | Path) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = model_dir / BUNDLE_FILENAME
    joblib.dump(bundle, bundle_path)

    with open(model_dir / METADATA_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(bundle.metadata(), fh, indent=2)

    return bundle_path


def load_bundle(model_dir: str | Path) -> ModelBundle:
    bundle_path = Path(model_dir) / BUNDLE_FILENAME
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"No trained model at {bundle_path}. Run `python scripts/train.py` first."
        )
    return joblib.load(bundle_path)


def read_metadata(model_dir: str | Path) -> dict | None:
    path = Path(model_dir) / METADATA_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
