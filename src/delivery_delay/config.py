"""Configuration loading.

Reads ``config/config.yaml`` and overlays a small set of environment variables
(see ``.env.example``). The loaded config is a plain nested dict wrapped in a
light ``Config`` accessor so callers can do ``cfg["models"]["eta_regressor"]``
or ``cfg.get("city.name")``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:  # optional: load a local .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


# Repo root = three levels up from this file (src/delivery_delay/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class Config:
    """Read-only dot/dict accessor over the parsed YAML config."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Look up a nested value with a dotted path, e.g. ``"city.name"``."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    # --- resolved, env-aware paths ---------------------------------------
    def path(self, key: str) -> Path:
        """Resolve a configured path relative to the project root."""
        rel = self.get(f"paths.{key}")
        if rel is None:
            raise KeyError(f"Unknown path key: {key}")
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def model_dir(self) -> Path:
        return Path(os.getenv("MODEL_DIR", str(self.path("model_dir"))))

    @property
    def seed(self) -> int:
        return int(self.get("project.random_seed", 42))


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Overlay env vars onto the config dict for the keys we expose."""
    if "paths" in data and os.getenv("MODEL_DIR"):
        data["paths"]["model_dir"] = os.environ["MODEL_DIR"]
    return data


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> Config:
    """Load (and cache) the project configuration."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    data = _apply_env_overrides(data)
    return Config(data)
