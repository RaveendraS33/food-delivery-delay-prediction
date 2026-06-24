"""Model persistence.

The trained artifact is a single ``ModelBundle`` (both models + the feature
column contract + metadata) saved with joblib, alongside a human-readable
``metadata.json``. Keeping the feature column list inside the bundle is what
lets the serving path reindex any incoming row to the exact training columns.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

BUNDLE_FILENAME = "model_bundle.joblib"
METADATA_FILENAME = "metadata.json"
VERSIONS_DIR = "versions"


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


def _version_tag(bundle: ModelBundle) -> str:
    """Derive a filesystem-safe version id from the training timestamp."""
    tag = re.sub(r"[^0-9T]", "", bundle.trained_at or "")
    return tag or "unversioned"


def save_bundle(bundle: ModelBundle, model_dir: str | Path) -> Path:
    """Persist the bundle.

    Writes the canonical ``model_bundle.joblib`` + ``metadata.json`` (the
    "latest" that serving loads), and also archives a timestamped copy under
    ``versions/`` so prior models are retained rather than overwritten.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = model_dir / BUNDLE_FILENAME
    joblib.dump(bundle, bundle_path)
    with open(model_dir / METADATA_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(bundle.metadata(), fh, indent=2)

    # Versioned archive (history of trained models).
    tag = _version_tag(bundle)
    versions = model_dir / VERSIONS_DIR
    versions.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, versions / f"model_bundle_{tag}.joblib")
    with open(versions / f"metadata_{tag}.json", "w", encoding="utf-8") as fh:
        json.dump(bundle.metadata(), fh, indent=2)

    logger.info("Saved model bundle %s (version %s)", bundle_path, tag)
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
