"""Server-side evaluation against the hidden reference route. Owner: Member 4.

The ONLY place in the API that reads .route. Results leave the server only as aggregate numbers.
If an evaluator trip has no .route file, metrics are returned as null (never a crash).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from api.schemas import Metrics
from src.data.loader import has_route, load_route
from src.eval.route_metrics import route_continuity, route_precision_recall, top1_from_preds
from src.labels.align_route import align


def score(trip_path: Path, out: dict, cfg: dict) -> Metrics:
    trip, g, res, clf, cands = out["trip"], out["graph"], out["hmm"], out["clf"], out["cands"]
    m = Metrics(n_route_breaks=res.n_breaks,
                route_continuity_hmm=_f(route_continuity(res.preds.pred_arc_id.to_numpy(), g.u, g.v,
                                                         res.preds.segment_id.to_numpy())))
    clf_full = trip.track[["obs_idx"]].merge(clf, on="obs_idx", how="left")
    m.route_continuity_classifier = _f(route_continuity(clf_full.pred_arc_id.fillna(-1).astype(np.int64).to_numpy(), g.u, g.v))
    if not has_route(trip_path):
        return m
    route = load_route(trip_path, len(trip.arcs))
    labels = align(trip, route, cfg)
    lab = labels[labels.label_arc_id >= 0]
    m.reference_available = True
    m.n_labelled_obs = int(len(lab))
    m.top1_accuracy_hmm = _f(top1_from_preds(res.preds, labels))
    m.top1_accuracy_classifier = _f(top1_from_preds(clf, labels))
    pr = route_precision_recall(res.path_arcs, route)
    m.route_precision, m.route_recall = _f(pr["route_precision"]), _f(pr["route_recall"])
    hit = cands.merge(lab[["obs_idx", "label_arc_id"]], on="obs_idx")
    found = hit[hit.arc_id == hit.label_arc_id].obs_idx.nunique()
    m.candidate_miss_rate = _f(1 - found / len(lab)) if len(lab) else None
    return m


def _f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)
