"""MOCK DATA ONLY - synthetic feature table that matches FEATURE_COLUMNS. Owner: Member 4.

Used to develop and test models, the CV runner and Optuna before the real features exist.
Numbers produced on this data are meaningless for the report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.schema import FEATURE_COLUMNS


def make_synthetic(n_trips: int = 20, obs_per_trip: int = 60, cands_per_obs: int = 8, seed: int = 42) -> pd.DataFrame:
    """One positive candidate per observation, with smaller distance/heading on average.

    Returns a DataFrame with KEY_COLUMNS + FEATURE_COLUMNS + y, labelled, label_arc_id
    (the same layout as data/processed/<partition>/features.parquet).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_trips):
        tid = f"{90000000 + t:08d}"
        for o in range(obs_per_trip):
            k = cands_per_obs
            pos = rng.integers(k)
            dist = np.sort(rng.gamma(2.0, 12.0, size=k))
            dist[pos] = max(0.2, dist[pos] * rng.uniform(0.2, 0.9))
            head = rng.uniform(0, 180, size=k)
            head[pos] = abs(rng.normal(0, 15))
            order = np.argsort(dist)
            rank = np.empty(k, int)
            rank[order] = np.arange(k)
            for c in range(k):
                arc = 1000 * t + 10 * c + o % 7
                rows.append({
                    "trip_id": tid, "obs_idx": o, "arc_id": arc,
                    "dist_m": dist[c], "dist_rank": rank[c], "dist_minus_min": dist[c] - dist.min(),
                    "heading_diff_deg": head[c], "heading_valid": 1, "speed_mps": rng.uniform(0, 20),
                    "dt_prev_s": 1.0, "arc_length_m": rng.uniform(10, 120), "proj_t": rng.uniform(),
                    "n_candidates": k, "u_degree": rng.integers(2, 8), "v_degree": rng.integers(2, 8),
                    "prev_obs_dist_same_arc": min(100, dist[c] + rng.normal(0, 3) ** 2),
                    "next_obs_dist_same_arc": min(100, dist[c] + rng.normal(0, 3) ** 2),
                    "reverse_exists": int(rng.uniform() < 0.9),
                    "y": int(c == pos), "labelled": True,
                })
    df = pd.DataFrame(rows)
    df["label_arc_id"] = df.groupby(["trip_id", "obs_idx"])["arc_id"].transform(lambda s: s[df.loc[s.index, "y"] == 1].iloc[0])
    for c in FEATURE_COLUMNS:
        df[c] = df[c].astype(np.float32)
    df["y"] = df["y"].astype(np.int8)
    return df


def synthetic_folds(df: pd.DataFrame, k: int = 5) -> dict:
    tids = sorted(df.trip_id.unique())
    return {"seed": 42, "n_splits": k, "folds": {t: i % k for i, t in enumerate(tids)}, "excluded": []}
