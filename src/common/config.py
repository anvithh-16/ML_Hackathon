"""Configuration helpers. Owner: Member 4."""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path

import yaml


def project_root() -> Path:
    """Repository root (folder that contains configs/base.yaml)."""
    env = os.environ.get("PATHSNAP_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=4)
def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Return a deep copy of configs/base.yaml (callers may mutate their copy)."""
    path = Path(path) if path else project_root() / "configs" / "base.yaml"
    return copy.deepcopy(_load_yaml(str(path)))


def load_model_config(name: str) -> dict:
    path = project_root() / "configs" / "models" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No model config {path}")
    return copy.deepcopy(_load_yaml(str(path)))


def load_hmm_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else project_root() / "configs" / "hmm.yaml"
    return copy.deepcopy(_load_yaml(str(path)))


def resolve(rel: str | os.PathLike) -> Path:
    """Resolve a path from the config relative to the repo root."""
    p = Path(rel)
    return p if p.is_absolute() else project_root() / p


def path_of(cfg: dict, key: str) -> Path:
    return resolve(cfg["paths"][key])
