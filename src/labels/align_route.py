"""Route-to-GPS alignment -> one label per GPS observation. Owner: Member 1.

The .route file is a chronological chain of arc ids with FAR fewer entries than GPS points,
so labels cannot come from row positions. We align points to POSITIONS in the route chain
with a monotonic dynamic programme (positions never go backwards). Because the state is the
route position (not the arc id), loops that revisit an arc are labelled correctly.

    cost(i, j)   = perpendicular distance of GPS point i to arc route[j]
    move j -> j' allowed if 0 <= j' - j <= max_skip, extra cost skip_penalty_m * (j' - j)
    points with no route position within search_m keep label -1 (DP carries over them)
    aligned points farther than max_dist_m from their arc get label -1
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.common.config import load_config, path_of
from src.common.logging_utils import get_logger
from src.common.schema import LABEL_COLUMNS, Trip
from src.data.geo import arc_endpoints, point_segment, project
from src.data.loader import list_trips, load_route, load_trip, trip_dir

log = get_logger("labels.align")
BIG = 1e9


def _route_position_candidates(trip: Trip, route: np.ndarray, search_m: float, step: float = 10.0):
    """For every GPS point: arrays (positions, distances) of route positions within search_m."""
    A, B = arc_endpoints(trip)
    uniq = np.unique(route)
    # densify unique route arcs
    L = np.hypot(*(B[uniq] - A[uniq]).T)
    nseg = np.maximum(1, np.ceil(L / step).astype(int))
    rep = np.repeat(np.arange(len(uniq)), nseg + 1)
    frac = np.concatenate([np.linspace(0, 1, k + 1) for k in nseg])
    pts = A[uniq][rep] + frac[:, None] * (B[uniq][rep] - A[uniq][rep])
    tree = cKDTree(pts)
    P = trip.track[["x", "y"]].to_numpy()
    hits = tree.query_ball_point(P, r=search_m + step)
    # arc -> list of route positions (loops give several)
    pos_of_arc: dict[int, list[int]] = {}
    for j, a in enumerate(route):
        pos_of_arc.setdefault(int(a), []).append(j)
    out = []
    for i, h in enumerate(hits):
        if not h:
            out.append((np.empty(0, np.int64), np.empty(0)))
            continue
        arcs_i = uniq[np.unique(rep[h])]
        d, _ = point_segment(np.repeat(P[i:i + 1], len(arcs_i), 0), A[arcs_i], B[arcs_i])
        keep = d <= search_m
        arcs_i, d = arcs_i[keep], d[keep]
        pos, dist = [], []
        for a, di in zip(arcs_i, d):
            for j in pos_of_arc[int(a)]:
                pos.append(j)
                dist.append(di)
        if pos:
            o = np.argsort(pos)
            out.append((np.asarray(pos, np.int64)[o], np.asarray(dist)[o]))
        else:
            out.append((np.empty(0, np.int64), np.empty(0)))
    return out


def align(trip: Trip, route: np.ndarray, cfg: dict | None = None) -> pd.DataFrame:
    """Return LABEL_COLUMNS, one row per observation, in obs_idx order."""
    if not trip.projected:
        trip = project(trip)
    cfg = cfg or load_config()
    ac = cfg["align"]
    n = len(trip.track)
    route = np.asarray(route, dtype=np.int64)
    labels = np.full(n, -1, np.int64)
    rpos = np.full(n, -1, np.int64)
    adist = np.full(n, np.nan)
    if len(route) == 0:
        return _frame(trip, labels, rpos, adist)

    cands = _route_position_candidates(trip, route, ac["search_m"])
    max_skip, pen = int(ac["max_skip"]), float(ac["skip_penalty_m"])

    states_pos, states_cost, back, idx_of = [], [], [], []  # per point that has states
    prev_pos = prev_cost = None
    for i in range(n):
        pos, dist = cands[i]
        if len(pos) == 0:
            continue
        if prev_pos is None:
            cost = dist.copy()
            bp = np.full(len(pos), -1, np.int64)
        else:
            diff = pos[:, None] - prev_pos[None, :]
            total = prev_cost[None, :] + pen * diff
            ok = (diff >= 0) & (diff <= max_skip)
            if not ok.any():                     # e.g. after a long GPS gap: allow a forward jump
                ok = diff >= 0
            if not ok.any():                     # cannot stay monotonic: restart with a big penalty
                ok = np.ones_like(diff, bool)
                total = total + BIG
            total = np.where(ok, total, np.inf)
            bp = np.argmin(total, axis=1)
            cost = total[np.arange(len(pos)), bp] + dist
        states_pos.append(pos)
        states_cost.append(cost)
        back.append(bp)
        idx_of.append(i)
        prev_pos, prev_cost = pos, cost

    if not idx_of:
        return _frame(trip, labels, rpos, adist)
    k = int(np.argmin(states_cost[-1]))
    for s in range(len(idx_of) - 1, -1, -1):
        i = idx_of[s]
        j = int(states_pos[s][k])
        d = float(cands[i][1][k])
        rpos[i] = j
        adist[i] = d
        if d <= ac["max_dist_m"]:
            labels[i] = route[j]
        k = int(back[s][k]) if s > 0 else -1
    return _frame(trip, labels, rpos, adist)


def _frame(trip, labels, rpos, adist) -> pd.DataFrame:
    df = pd.DataFrame({
        "trip_id": trip.trip_id,
        "obs_idx": trip.track["obs_idx"].to_numpy(),
        "label_arc_id": labels,
        "route_pos": rpos,
        "align_dist_m": adist,
    })
    return df[LABEL_COLUMNS]


def align_partition(partition: str, cfg: dict | None = None, limit: int | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    root = path_of(cfg, "data_root")
    out = []
    for tid in list_trips(root, partition)[:limit]:
        d = trip_dir(root, partition, tid)
        trip = project(load_trip(d, partition))
        out.append(align(trip, load_route(d, len(trip.arcs)), cfg))
        log.info("aligned %s %s: %.1f%% labelled", partition, tid, 100 * (out[-1].label_arc_id >= 0).mean())
    df = pd.concat(out, ignore_index=True)
    dest = path_of(cfg, "processed") / partition
    dest.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest / "labels.parquet", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Align GPS observations to the reference route")
    ap.add_argument("--partition", default="train", choices=["train", "student_test"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    df = align_partition(a.partition, limit=a.limit)
    print(f"{a.partition}: {len(df)} observations, {100 * (df.label_arc_id >= 0).mean():.2f}% labelled, "
          f"median align distance {df.align_dist_m.median():.2f} m")


if __name__ == "__main__":
    main()
