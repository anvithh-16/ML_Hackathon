"""Prediction pipeline used by the API. Owner: Member 4.

loader -> projection -> candidates -> features -> model scores -> HMM/Viterbi -> lat/lon output.
This module must NEVER read the reference route (.route); tests/test_api.py greps for it.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.candidates.generate import generate
from src.common.schema import FEATURE_COLUMNS
from src.data.geo import project, to_latlon
from src.data.loader import load_trip
from src.features.build_features import build
from src.hmm.decode import classifier_only, decode_trip
from src.hmm.graph import RoadGraph


def path_latlon(trip, path_arcs: list[list[int]]) -> list[list[list[float]]]:
    lat, lon = trip.nodes.lat.to_numpy(), trip.nodes.lon.to_numpy()
    u, v = trip.arcs.u.to_numpy(), trip.arcs.v.to_numpy()
    lines = []
    for seg in path_arcs:
        if not seg:
            continue
        pts: list[list[float]] = []
        for a in seg:
            start = [float(lat[u[a]]), float(lon[u[a]])]
            if not pts or pts[-1] != start:
                pts.append(start)
            pts.append([float(lat[v[a]]), float(lon[v[a]])])
        lines.append(pts)
    return lines


def predict_trip(trip_path: Path, model, cfg: dict, hmm_cfg: dict, max_points: int = 3000) -> dict:
    """model=None -> distance-only HMM (no trained model). Returns everything the scorer and response need."""
    tm = {}
    t0 = time.perf_counter()
    trip = project(load_trip(trip_path, "eval"))
    tm["load_project_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cands = generate(trip, cfg)
    tm["candidates_s"] = time.perf_counter() - t0

    scores = None
    if model is not None:
        t0 = time.perf_counter()
        feats = build(trip, cands, None, cfg)
        tm["features_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        scores = model.score_candidates(feats[["trip_id", "obs_idx", "arc_id"]], feats[FEATURE_COLUMNS])
        tm["model_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    graph = RoadGraph(trip)
    res = decode_trip(trip, cands, scores, hmm_cfg, cfg, graph)
    tm["hmm_s"] = time.perf_counter() - t0
    clf = classifier_only(cands, scores, sigma_m=hmm_cfg["sigma_m"])

    tr = trip.track
    step = max(1, int(np.ceil(len(tr) / max_points)))
    gps = [[float(a), float(b)] for a, b in zip(tr.lat.to_numpy()[::step], tr.lon.to_numpy()[::step])]
    return {"trip": trip, "cands": cands, "graph": graph, "hmm": res, "clf": clf, "gps_latlon": gps,
            "route_latlon": path_latlon(trip, res.path_arcs), "timings": tm}
