"""Common model interface for all five candidate classifiers. Owner: Member 4.

Every model:  takes X = df[FEATURE_COLUMNS] (exact order), y in {0,1};
returns P(candidate is the correct arc) as a 1-D array clipped to [1e-6, 1-1e-6].
Model-specific preprocessing (e.g. scaling) lives INSIDE build_estimator() as a Pipeline.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.common.schema import FEATURE_COLUMNS, FEATURE_VERSION, PROBA_CLIP, SCORE_COLUMNS, SchemaError

PREDICT_CHUNK = 100_000


def _clean_params(params: dict) -> dict:
    """YAML cannot express Python None inside categorical choices -> we use the string 'null'."""
    return {k: (None if v in ("null", "None") else v) for k, v in (params or {}).items()}


class BaseCandidateModel:
    name: str = "base"
    family: str = "unknown"
    needs_scaling: bool = False
    proba_quality: str = "native"  # "native" or "coarse" (vote fractions / leaf frequencies)

    def __init__(self, params: dict | None = None, seed: int = 42, calibration: str = "none"):
        self.params = _clean_params(params or {})
        self.seed = int(seed)
        self.calibration = calibration
        self.estimator_ = None
        self.train_time_s: float | None = None
        self.n_train_rows: int | None = None

    # ------------------------------------------------------------- subclass hook
    def build_estimator(self):
        raise NotImplementedError

    # ------------------------------------------------------------- helpers
    @staticmethod
    def check_X(X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise SchemaError("X must be a DataFrame with FEATURE_COLUMNS")
        cols = list(X.columns)
        if cols != FEATURE_COLUMNS:
            missing = [c for c in FEATURE_COLUMNS if c not in cols]
            extra = [c for c in cols if c not in FEATURE_COLUMNS]
            raise SchemaError(f"X columns differ from FEATURE_COLUMNS. missing={missing} extra={extra} "
                              f"(pass df[FEATURE_COLUMNS])")
        return X.to_numpy(dtype=np.float32)

    # ------------------------------------------------------------- API
    def fit(self, X: pd.DataFrame, y, groups=None) -> "BaseCandidateModel":
        Xa = self.check_X(X)
        y = np.asarray(y).astype(int)
        if set(np.unique(y)) - {0, 1}:
            raise ValueError("y must be binary 0/1")
        if len(np.unique(y)) < 2:
            raise ValueError("y contains a single class - check labels / filters")
        est = self.build_estimator()
        if self.calibration in ("sigmoid", "isotonic"):
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.model_selection import GroupKFold
            if groups is None:
                raise ValueError("calibration needs groups (trip ids) for group-aware CV")
            cv = list(GroupKFold(n_splits=3).split(Xa, y, groups))
            est = CalibratedClassifierCV(est, method=self.calibration, cv=cv)
        t0 = time.perf_counter()
        est.fit(Xa, y)
        self.train_time_s = time.perf_counter() - t0
        self.n_train_rows = len(y)
        self.estimator_ = est
        return self

    def _raw_proba(self, Xa: np.ndarray) -> np.ndarray:
        return self.estimator_.predict_proba(Xa)[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError(f"{self.name}: call fit() or load() first")
        Xa = self.check_X(X)
        out = np.empty(len(Xa), dtype=np.float64)
        for s in range(0, len(Xa), PREDICT_CHUNK):
            out[s:s + PREDICT_CHUNK] = self._raw_proba(Xa[s:s + PREDICT_CHUNK])
        if not np.isfinite(out).all():
            raise ValueError(f"{self.name}: non-finite probabilities")
        return np.clip(out, PROBA_CLIP, 1 - PROBA_CLIP)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int8)

    def score_candidates(self, keys: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        s = keys[["trip_id", "obs_idx", "arc_id"]].copy()
        s["score"] = self.predict_proba(X)
        return s[SCORE_COLUMNS].reset_index(drop=True)

    def get_config(self) -> dict:
        return {"name": self.name, "family": self.family, "params": self.params, "seed": self.seed,
                "calibration": self.calibration, "feature_version": FEATURE_VERSION,
                "feature_columns": FEATURE_COLUMNS, "train_time_s": self.train_time_s,
                "n_train_rows": self.n_train_rows, "sklearn": sklearn.__version__,
                "python": platform.python_version()}

    def save(self, out_dir) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator_, out / "model.joblib", compress=3)
        cfg = self.get_config()
        (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
        return out

    @classmethod
    def load(cls, out_dir) -> "BaseCandidateModel":
        out = Path(out_dir)
        cfg = json.loads((out / "config.json").read_text())
        if cfg.get("feature_version") != FEATURE_VERSION:
            raise SchemaError(f"artifact {out} was trained on features {cfg.get('feature_version')}, "
                              f"current is {FEATURE_VERSION}: retrain")
        if cls is BaseCandidateModel:
            from src.models.registry import get_model_class
            cls = get_model_class(cfg["name"])
        m = cls(cfg.get("params"), cfg.get("seed", 42), cfg.get("calibration", "none"))
        m.estimator_ = joblib.load(out / "model.joblib")
        m.train_time_s = cfg.get("train_time_s")
        m.n_train_rows = cfg.get("n_train_rows")
        return m
