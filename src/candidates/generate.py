"""Candidate road segments for each GPS observation. Owner: Member 1.

Arcs are densified every `densify_step_m` into a KD-tree so long arcs are found even if their
midpoint is far away. Exact point-to-segment distance decides membership:
  * all arcs within radius_m;
  * if fewer than k_fallback, the nearest arcs up to fallback_max_m;
  * at most max_per_obs, nearest first (cand_rank 0 = nearest).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.common.config import load_config, path_of
from src.common.logging_utils import get_logger
from src.common.schema import CAND_COLUMNS, Trip
from src.data.geo import arc_endpoints, point_segment, project

log = get_logger("candidates")


class ArcIndex:
    """Spatial index over densified arcs of one trip (reuse it for many queries)."""

    def __init__(self, trip: Trip, step: float):
        self.A, self.B = arc_endpoints(trip)
        L = np.hypot(*(self.B - self.A).T)
        nseg = np.maximum(1, np.ceil(L / step).astype(int))
        self.rep = np.repeat(np.arange(len(L)), nseg + 1)
        frac = np.concatenate([np.linspace(0, 1, k + 1) for k in nseg])
        pts = self.A[self.rep] + frac[:, None] * (self.B[self.rep] - self.A[self.rep])
        self.tree = cKDTree(pts)
        self.step = step

    def within(self, P: np.ndarray, r: float):
        """Pairs (point_index, arc_id, dist, t) for arcs within r of each point (exact distance)."""
        hits = self.tree.query_ball_point(P, r=r + self.step)
        lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
        if lens.sum() == 0:
            return (np.empty(0, np.int64),) * 2 + (np.empty(0),) * 2
        pi = np.repeat(np.arange(len(P)), lens)
        ai = self.rep[np.concatenate([np.asarray(h, np.int64) for h in hits if h])]
        pair = np.unique(pi * (len(self.A) + 1) + ai)
        pi, ai = pair // (len(self.A) + 1), pair % (len(self.A) + 1)
        d, t = point_segment(P[pi], self.A[ai], self.B[ai])
        keep = d <= r
        return pi[keep], ai[keep], d[keep], t[keep]

    def nearest(self, P: np.ndarray, k: int, max_m: float):
        """k nearest distinct arcs within max_m for each point."""
        kk = min(len(self.rep), max(8 * k, 32))
        dd, ii = self.tree.query(P, k=kk, distance_upper_bound=max_m + self.step)
        out_p, out_a = [], []
        for i in range(len(P)):
            valid = np.isfinite(dd[i])
            arcs = pd.unique(self.rep[ii[i][valid]])
            out_p.append(np.full(len(arcs), i))
            out_a.append(arcs)
        if not out_p:
            return (np.empty(0, np.int64),) * 2 + (np.empty(0),) * 2
        pi, ai = np.concatenate(out_p).astype(np.int64), np.concatenate(out_a).astype(np.int64)
        d, t = point_segment(P[pi], self.A[ai], self.B[ai])
        keep = d <= max_m
        return pi[keep], ai[keep], d[keep], t[keep]


def generate(trip: Trip, cfg: dict | None = None, index: ArcIndex | None = None) -> pd.DataFrame:
    """CAND_COLUMNS rows sorted by (obs_idx, cand_rank). Observations may have zero rows."""
    if not trip.projected:
        trip = project(trip)
    cc = (cfg or load_config())["candidates"]
    index = index or ArcIndex(trip, cc["densify_step_m"])
    P = trip.track[["x", "y"]].to_numpy()
    pi, ai, d, t = index.within(P, cc["radius_m"])

    counts = np.bincount(pi, minlength=len(P))
    need = np.where(counts < cc["k_fallback"])[0]
    if len(need):
        fp, fa, fd, ft = index.nearest(P[need], cc["k_fallback"], cc["fallback_max_m"])
        fp = need[fp]
        df_f = pd.DataFrame({"p": fp, "a": fa, "d": fd, "t": ft}).sort_values(["p", "d"])
        df_f = df_f.groupby("p", sort=False).head(cc["k_fallback"])
        pi = np.concatenate([pi, df_f.p.to_numpy()])
        ai = np.concatenate([ai, df_f.a.to_numpy()])
        d = np.concatenate([d, df_f.d.to_numpy()])
        t = np.concatenate([t, df_f.t.to_numpy()])

    df = pd.DataFrame({"obs_idx": pi, "arc_id": ai, "dist_m": d, "proj_t": t})
    df = df.drop_duplicates(["obs_idx", "arc_id"]).sort_values(["obs_idx", "dist_m", "arc_id"], kind="stable")
    df = df.groupby("obs_idx", sort=False).head(cc["max_per_obs"]).reset_index(drop=True)
    df["cand_rank"] = df.groupby("obs_idx").cumcount().astype(np.int64)
    df["obs_idx"] = trip.track["obs_idx"].to_numpy()[df["obs_idx"].to_numpy()]
    df.insert(0, "trip_id", trip.trip_id)
    df["dist_m"] = df["dist_m"].astype(np.float64)
    return df[CAND_COLUMNS]


def radius_sweep(radii=(30, 50, 75, 100), partition: str = "train", limit: int | None = None) -> pd.DataFrame:
    """Candidate miss rate vs mean candidate count per radius (TRAIN only). Needs labels.parquet."""
    from src.data.loader import list_trips, load_trip, trip_dir

    cfg = load_config()
    root = path_of(cfg, "data_root")
    labels = pd.read_parquet(path_of(cfg, "processed") / partition / "labels.parquet")
    labels["trip_id"] = labels["trip_id"].astype(str)
    rows = []
    for tid in list_trips(root, partition)[:limit]:
        if tid in cfg.get("exclude_trips_train", []):
            continue
        trip = project(load_trip(trip_dir(root, partition, tid), partition))
        idx = ArcIndex(trip, cfg["candidates"]["densify_step_m"])
        lab = labels[labels.trip_id == tid].set_index("obs_idx")["label_arc_id"]
        for r in radii:
            c = dict(cfg)
            c["candidates"] = dict(cfg["candidates"], radius_m=r)
            cand = generate(trip, c, idx)
            m = cand.merge(lab.rename("lab"), left_on="obs_idx", right_index=True)
            hit = m[m.arc_id == m.lab].obs_idx.nunique()
            n_lab = int((lab >= 0).sum())
            rows.append({"trip_id": tid, "radius_m": r, "n_labelled": n_lab, "n_hit": hit,
                         "n_cand_rows": len(cand), "n_obs": len(trip.track)})
    df = pd.DataFrame(rows)
    g = df.groupby("radius_m").sum(numeric_only=True)
    summary = pd.DataFrame({
        "radius_m": g.index,
        "candidate_miss_rate": 1 - g.n_hit / g.n_labelled,
        "mean_candidates_per_obs": g.n_cand_rows / g.n_obs,
    }).reset_index(drop=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Candidate generation utilities")
    ap.add_argument("--sweep", action="store_true", help="write results/candidates/radius_sweep.csv")
    ap.add_argument("--limit", type=int, default=None, help="only the first N train trips")
    a = ap.parse_args()
    if a.sweep:
        s = radius_sweep(limit=a.limit)
        out = path_of(load_config(), "results") / "candidates"
        out.mkdir(parents=True, exist_ok=True)
        s.to_csv(out / "radius_sweep.csv", index=False)
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()
