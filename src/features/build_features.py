"""Feature table for (observation, candidate) pairs. Owner: Member 1.

The ONLY feature implementation used by all five models and by the API. Nothing here reads
the route: labels are joined at the end only when they are passed in.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.common.config import load_config, path_of
from src.common.logging_utils import get_logger
from src.common.schema import (CAND_COLUMNS, FEATURE_COLUMNS, KEY_COLUMNS, LABEL_COLUMNS, Trip,
                               check_columns)
from src.data.geo import arc_endpoints, point_segment, project

log = get_logger("features")


def _obs_motion(trip: Trip, fc: dict):
    """Per-observation heading (deg), heading_valid, speed (m/s), dt_prev (s)."""
    x, y, t = (trip.track[c].to_numpy() for c in ("x", "y", "t"))
    n = len(x)
    prev = np.r_[0, np.arange(n - 1)]
    nxt = np.r_[np.arange(1, n), n - 1]
    dx, dy = x[nxt] - x[prev], y[nxt] - y[prev]
    disp = np.hypot(dx, dy)
    dt = np.maximum(t[nxt] - t[prev], 1e-3)
    heading = np.degrees(np.arctan2(dx, dy)) % 360.0
    valid = (disp >= fc["heading_min_disp_m"]).astype(np.int8)
    speed = np.where(n > 1, disp / dt, 0.0)
    dt_prev = np.r_[0.0, np.diff(t)]
    return heading, valid, speed, np.minimum(dt_prev, fc["dt_cap_s"])


def build(trip: Trip, cands: pd.DataFrame, labels: pd.DataFrame | None = None,
          cfg: dict | None = None) -> pd.DataFrame:
    """KEY_COLUMNS + FEATURE_COLUMNS (+ y, labelled, label_arc_id when labels are given)."""
    if not trip.projected:
        trip = project(trip)
    check_columns(cands, CAND_COLUMNS, "candidates")
    cfg = cfg or load_config()
    fc = cfg["features"]
    cap = float(fc["neighbour_dist_cap_m"])

    c = cands.sort_values(["obs_idx", "cand_rank"], kind="stable").reset_index(drop=True)
    obs = c["obs_idx"].to_numpy()
    arc = c["arc_id"].to_numpy()
    arcs = trip.arcs
    heading, hvalid, speed, dt_prev = _obs_motion(trip, fc)

    bearing = arcs["bearing_deg"].to_numpy()[arc]
    diff = np.abs((heading[obs] - bearing + 180.0) % 360.0 - 180.0)
    hv = hvalid[obs]

    n_nodes = len(trip.nodes)
    u, v = arcs["u"].to_numpy(), arcs["v"].to_numpy()
    deg = np.bincount(u, minlength=n_nodes) + np.bincount(v, minlength=n_nodes)

    grp_min = c.groupby("obs_idx")["dist_m"].transform("min").to_numpy()
    n_cand = c.groupby("obs_idx")["obs_idx"].transform("size").to_numpy()

    A, B = arc_endpoints(trip)
    P = trip.track[["x", "y"]].to_numpy()
    n = len(P)
    has_prev = obs > 0
    has_next = obs < n - 1
    prev_d = np.full(len(c), cap)
    next_d = np.full(len(c), cap)
    if has_prev.any():
        d, _ = point_segment(P[obs[has_prev] - 1], A[arc[has_prev]], B[arc[has_prev]])
        prev_d[has_prev] = np.minimum(d, cap)
    if has_next.any():
        d, _ = point_segment(P[obs[has_next] + 1], A[arc[has_next]], B[arc[has_next]])
        next_d[has_next] = np.minimum(d, cap)

    f = pd.DataFrame({
        "trip_id": c["trip_id"].astype(str).to_numpy(),
        "obs_idx": obs.astype(np.int64),
        "arc_id": arc.astype(np.int64),
        "dist_m": c["dist_m"].to_numpy(),
        "dist_rank": c["cand_rank"].to_numpy(),
        "dist_minus_min": c["dist_m"].to_numpy() - grp_min,
        "heading_diff_deg": np.where(hv == 1, diff, 90.0),
        "heading_valid": hv,
        "speed_mps": speed[obs],
        "dt_prev_s": dt_prev[obs],
        "arc_length_m": arcs["length_m"].to_numpy()[arc],
        "proj_t": c["proj_t"].to_numpy(),
        "n_candidates": n_cand,
        "u_degree": deg[u[arc]],
        "v_degree": deg[v[arc]],
        "prev_obs_dist_same_arc": prev_d,
        "next_obs_dist_same_arc": next_d,
        "reverse_exists": (arcs["reverse_arc_id"].to_numpy()[arc] >= 0).astype(np.int8),
    })
    for col in FEATURE_COLUMNS:
        f[col] = f[col].astype(np.float32)
    f = f[KEY_COLUMNS + FEATURE_COLUMNS]
    if not np.isfinite(f[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError(f"{trip.trip_id}: non-finite feature values")

    if labels is not None:
        check_columns(labels, LABEL_COLUMNS, "labels")
        lab = labels.set_index("obs_idx")["label_arc_id"]
        f["label_arc_id"] = lab.reindex(f["obs_idx"]).fillna(-1).astype(np.int64).to_numpy()
        f["labelled"] = f["label_arc_id"] >= 0
        f["y"] = (f["arc_id"] == f["label_arc_id"]).astype(np.int8)
        f = f[KEY_COLUMNS + FEATURE_COLUMNS + ["y", "labelled", "label_arc_id"]]
    return f


def process_trip(root, partition: str, tid: str, cfg: dict):
    """Labels, candidates and features for one trip (used by build_partition)."""
    from src.candidates.generate import generate
    from src.data.loader import load_route, load_trip, trip_dir
    from src.labels.align_route import align

    d = trip_dir(root, partition, tid)
    trip = project(load_trip(d, partition))
    labels = align(trip, load_route(d, len(trip.arcs)), cfg)
    cands = generate(trip, cfg)
    feats = build(trip, cands, labels, cfg)
    return labels, cands, feats


def build_partition(partition: str, limit: int | None = None, n_jobs: int = -1, cfg: dict | None = None) -> dict:
    from src.data.loader import list_trips

    cfg = cfg or load_config()
    root = path_of(cfg, "data_root")
    tids = list_trips(root, partition)[:limit]
    log.info("building %s: %d trips", partition, len(tids))
    res = Parallel(n_jobs=n_jobs, verbose=0)(delayed(process_trip)(root, partition, t, cfg) for t in tids)
    out = path_of(cfg, "processed") / partition
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.concat([r[0] for r in res], ignore_index=True)
    cands = pd.concat([r[1] for r in res], ignore_index=True)
    feats = pd.concat([r[2] for r in res], ignore_index=True)
    labels.to_parquet(out / "labels.parquet", index=False)
    cands.to_parquet(out / "candidates.parquet", index=False)
    feats.to_parquet(out / "features.parquet", index=False)
    obs_lab = labels.label_arc_id >= 0
    hit = feats[feats.y == 1][["trip_id", "obs_idx"]].drop_duplicates()
    summary = {
        "partition": partition, "trips": len(tids), "observations": len(labels),
        "labelled_obs": int(obs_lab.sum()), "candidate_rows": len(feats),
        "candidate_miss_rate_labelled": float(1 - len(hit) / max(int(obs_lab.sum()), 1)),
        "feature_version": cfg["features"]["version"],
    }
    log.info("summary %s", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build labels, candidates and features for a partition")
    ap.add_argument("--partition", default="train", choices=["train", "student_test"])
    ap.add_argument("--limit", type=int, default=None, help="first N trips only (features v0)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    s = build_partition(a.partition, a.limit, a.n_jobs)
    for k, v in s.items():
        print(f"{k:32s} {v}")


if __name__ == "__main__":
    main()
