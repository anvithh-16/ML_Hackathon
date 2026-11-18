"""Decode a whole trip into a connected route. Owner: Member 3.

python -m src.hmm.decode --trip 00000001 --partition train --emission distance
python -m src.hmm.decode --trip 00000000 --partition student_test --scores results/scores/xgb__default/student_test.parquet

Segments: a new segment starts when dt > gap_split_s, the straight jump > gap_split_m,
the observation has no candidates (prediction -1), or no finite transition exists (BrokenChain).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.common.config import load_config, load_hmm_config, path_of
from src.common.logging_utils import get_logger
from src.common.schema import PRED_COLUMNS, Trip
from src.data.geo import project
from src.hmm.graph import RoadGraph
from src.hmm.viterbi import (BrokenChain, backtrack, distance_scores, emission_logprob,
                             transition_logprob, viterbi_step)

log = get_logger("hmm.decode")


@dataclass
class DecodeResult:
    preds: pd.DataFrame                      # PRED_COLUMNS, one row per observation
    path_arcs: list[list[int]]               # connected arc sequence per segment
    n_breaks: int = 0
    timings: dict = field(default_factory=dict)


def _obs_groups(trip: Trip, cands: pd.DataFrame, scores: pd.DataFrame | None, hcfg: dict):
    """Per observation (in time order): arrays arc_ids, proj_t, score."""
    c = cands[["obs_idx", "arc_id", "dist_m", "proj_t"]]
    if scores is not None:
        s = scores[["obs_idx", "arc_id", "score"]]
        c = c.merge(s, on=["obs_idx", "arc_id"], how="inner")
    else:
        c = c.assign(score=distance_scores(c["dist_m"].to_numpy(), hcfg["sigma_m"]))
    c = c.sort_values(["obs_idx", "dist_m"], kind="stable")
    c = c.groupby("obs_idx", sort=False).head(hcfg.get("max_candidates", 30))
    groups = {}
    for o, g in c.groupby("obs_idx", sort=True):
        groups[int(o)] = (g.arc_id.to_numpy(np.int64), g.proj_t.to_numpy(float), g.score.to_numpy(float))
    return groups


def decode_trip(trip: Trip, cands: pd.DataFrame, scores: pd.DataFrame | None = None,
                hmm_cfg: dict | None = None, cfg: dict | None = None, graph: RoadGraph | None = None,
                obs_subset: np.ndarray | None = None) -> DecodeResult:
    """scores=None -> distance-only emissions. obs_subset: decode only these obs_idx (e.g. strided OOF)."""
    t0 = time.perf_counter()
    if not trip.projected:
        trip = project(trip)
    cfg = cfg or load_config()
    h = hmm_cfg or load_hmm_config()
    g = graph or RoadGraph(trip)
    groups = _obs_groups(trip, cands, scores, h)
    tr = trip.track
    obs_all = tr["obs_idx"].to_numpy()
    if obs_subset is not None:
        keep = np.isin(obs_all, obs_subset)
        obs_all = obs_all[keep]
    xs, ys, ts = (tr.set_index("obs_idx").loc[obs_all, c].to_numpy() for c in ("x", "y", "t"))
    stride = max(1, int(h.get("decode_stride", 1)))

    pred = np.full(len(obs_all), -1, np.int64)
    seg = np.full(len(obs_all), -1, np.int64)
    paths: list[list[int]] = []
    n_breaks = 0
    seg_id = -1

    # current segment state
    cur_idx: list[int] = []        # positions in obs_all decoded in this segment
    cur_states: list[tuple] = []   # (arcs, t) per decoded position
    bps: list[np.ndarray] = []
    delta = None

    def close_segment():
        nonlocal delta, cur_idx, cur_states, bps
        if delta is None or not cur_idx:
            delta, cur_idx, cur_states, bps = None, [], [], []
            return
        ks = backtrack(delta, bps)
        arcs_seq = [int(cur_states[j][0][k]) for j, k in enumerate(ks)]
        for j, pos in enumerate(cur_idx):
            pred[pos] = arcs_seq[j]
            seg[pos] = seg_id
        path = [arcs_seq[0]]
        for a, b in zip(arcs_seq[:-1], arcs_seq[1:]):
            if a == b:
                continue
            path.extend(g.shortest_arc_path(a, b, limit_m=2000.0))
            path.append(b)
        paths.append(path)
        delta, cur_idx, cur_states, bps = None, [], [], []

    last_pos = None
    for pos in range(len(obs_all)):
        o = int(obs_all[pos])
        grp = groups.get(o)
        new_seg = False
        if last_pos is not None:
            dt = ts[pos] - ts[last_pos]
            jump = np.hypot(xs[pos] - xs[last_pos], ys[pos] - ys[last_pos])
            new_seg = dt > cfg["gap_split_s"] or jump > cfg["gap_split_m"]
        if grp is None:                      # no candidates: prediction -1, break the chain
            if delta is not None:
                n_breaks += 1
            close_segment()
            last_pos = None
            continue
        # stride: inside a running segment, only decode every k-th point (others filled later)
        if stride > 1 and delta is not None and not new_seg and (pos - cur_idx[-1]) < stride and pos != len(obs_all) - 1:
            continue
        arcs, tt, sc = grp
        log_e = emission_logprob(sc, h["emission_eps"])
        if delta is None or new_seg:
            if delta is not None:
                n_breaks += 1
            close_segment()
            seg_id += 1
            delta = log_e
            cur_idx, cur_states, bps = [pos], [(arcs, tt)], []
            last_pos = pos
            continue
        pa, pt = cur_states[-1]
        prev = cur_idx[-1]
        gc = float(np.hypot(xs[pos] - xs[prev], ys[pos] - ys[prev]))
        limit = h["dijkstra_limit_factor"] * gc + h["dijkstra_limit_extra_m"]
        net = np.empty((len(pa), len(arcs)))
        uturn = np.zeros((len(pa), len(arcs)), bool)
        for i, (a, ta) in enumerate(zip(pa, pt)):
            for j, (b, tb) in enumerate(zip(arcs, tt)):
                net[i, j] = g.network_distance(int(a), float(ta), int(b), float(tb), limit)
                uturn[i, j] = g.rev[a] == b
        log_t = transition_logprob(net, gc, h["beta_m"], uturn, h["uturn_penalty"])
        try:
            delta, bp = viterbi_step(delta, log_t, log_e)
            bps.append(bp)
            cur_idx.append(pos)
            cur_states.append((arcs, tt))
        except BrokenChain:
            n_breaks += 1
            close_segment()
            seg_id += 1
            delta = log_e
            cur_idx, cur_states, bps = [pos], [(arcs, tt)], []
        last_pos = pos
    close_segment()

    # fill observations skipped by the stride with the nearest decoded neighbour in the same segment
    if stride > 1:
        dec = np.where(pred >= 0)[0]
        for pos in np.where(pred < 0)[0]:
            if int(obs_all[pos]) not in groups or len(dec) == 0:
                continue
            j = dec[np.argmin(np.abs(ts[dec] - ts[pos]))]
            pred[pos], seg[pos] = pred[j], seg[j]

    preds = pd.DataFrame({"trip_id": trip.trip_id, "obs_idx": obs_all, "pred_arc_id": pred, "segment_id": seg})
    return DecodeResult(preds=preds[PRED_COLUMNS], path_arcs=paths, n_breaks=n_breaks,
                        timings={"decode_s": time.perf_counter() - t0})


def classifier_only(cands: pd.DataFrame, scores: pd.DataFrame | None, trip_id: str | None = None,
                    sigma_m: float = 5.0) -> pd.DataFrame:
    """Independent argmax per observation (ties -> smaller dist_m). scores=None -> nearest candidate."""
    c = cands[["trip_id", "obs_idx", "arc_id", "dist_m"]]
    if scores is not None:
        c = c.merge(scores[["obs_idx", "arc_id", "score"]], on=["obs_idx", "arc_id"], how="inner")
    else:
        c = c.assign(score=distance_scores(c["dist_m"].to_numpy(), sigma_m))
    c = c.sort_values(["obs_idx", "score", "dist_m"], ascending=[True, False, True], kind="mergesort")
    top = c.drop_duplicates("obs_idx")
    out = pd.DataFrame({"trip_id": top.trip_id.astype(str), "obs_idx": top.obs_idx, "pred_arc_id": top.arc_id,
                        "segment_id": 0})
    return out[PRED_COLUMNS].reset_index(drop=True)


def main() -> None:
    from src.data.loader import load_trip, trip_dir
    from src.candidates.generate import generate

    ap = argparse.ArgumentParser(description="Decode one trip with the HMM")
    ap.add_argument("--trip", required=True)
    ap.add_argument("--partition", default="train")
    ap.add_argument("--emission", choices=["distance", "scores"], default="distance")
    ap.add_argument("--scores", help="score parquet (SCORE_COLUMNS) when --emission scores")
    a = ap.parse_args()
    cfg = load_config()
    trip = project(load_trip(trip_dir(path_of(cfg, "data_root"), a.partition, a.trip), a.partition))
    cands = generate(trip, cfg)
    scores = None
    if a.scores:
        s = pd.read_parquet(a.scores)
        s["trip_id"] = s.trip_id.astype(str)
        scores = s[s.trip_id == a.trip]
    res = decode_trip(trip, cands, scores, cfg=cfg)
    print(f"trip {a.trip}: {len(res.preds)} obs, {res.preds.segment_id.max() + 1} segments, "
          f"{res.n_breaks} breaks, {(res.preds.pred_arc_id < 0).sum()} unmatched, "
          f"{sum(len(p) for p in res.path_arcs)} path arcs, {res.timings['decode_s']:.2f}s")


if __name__ == "__main__":
    main()
