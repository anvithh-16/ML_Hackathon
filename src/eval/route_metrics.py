"""Route-level metrics. Owner: Member 3 (used by M2's compare_models and M4's API scorer)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def route_continuity(pred_arc_ids, u: np.ndarray, v: np.ndarray, segment_ids=None) -> float:
    """Share of consecutive observation pairs (same segment, both predicted) whose predicted arcs are
    equal or connected (end node of the first == start node of the second)."""
    p = np.asarray(pred_arc_ids, np.int64)
    if len(p) < 2:
        return float("nan")
    a, b = p[:-1], p[1:]
    ok = (a >= 0) & (b >= 0)
    if segment_ids is not None:
        s = np.asarray(segment_ids)
        ok &= s[:-1] == s[1:]
    if not ok.any():
        return float("nan")
    a, b = a[ok], b[ok]
    return float(np.mean((a == b) | (v[a] == u[b])))


def path_continuity(path_arcs: list[list[int]], u: np.ndarray, v: np.ndarray) -> float:
    """Continuity of the reconstructed path itself (consecutive path arcs connected)."""
    pairs = [(a, b) for seg in path_arcs for a, b in zip(seg[:-1], seg[1:])]
    if not pairs:
        return float("nan")
    a, b = np.array(pairs).T
    return float(np.mean(v[a] == u[b]))


def top1_from_preds(preds: pd.DataFrame, labels: pd.DataFrame) -> float:
    m = labels[labels.label_arc_id >= 0].merge(preds[["obs_idx", "pred_arc_id"]], on="obs_idx", how="left")
    if len(m) == 0:
        return float("nan")
    return float((m.pred_arc_id.fillna(-1).astype(np.int64) == m.label_arc_id).mean())


def route_precision_recall(path_arcs, route: np.ndarray, lengths: np.ndarray | None = None) -> dict:
    """Overlap of predicted path arcs with reference route arcs (set-based, and length-weighted)."""
    if isinstance(path_arcs, list) and path_arcs and isinstance(path_arcs[0], (list, tuple)):
        P = set(int(a) for seg in path_arcs for a in seg)          # reconstructed path per segment
    else:
        P = set(int(a) for a in np.asarray(path_arcs).ravel() if a >= 0)   # per-observation predictions
    R = set(int(a) for a in route)
    inter = P & R
    out = {"route_precision": len(inter) / len(P) if P else float("nan"),
           "route_recall": len(inter) / len(R) if R else float("nan")}
    if lengths is not None and P and R:
        L = lambda S: float(sum(lengths[a] for a in S))
        out["route_precision_len"] = L(inter) / max(L(P), 1e-9)
        out["route_recall_len"] = L(inter) / max(L(R), 1e-9)
    return out


def summarise(preds: pd.DataFrame, path_arcs, labels: pd.DataFrame | None, route: np.ndarray | None,
              u: np.ndarray, v: np.ndarray, lengths: np.ndarray, n_breaks: int = 0) -> dict:
    out = {"route_continuity": route_continuity(preds.pred_arc_id.to_numpy(), u, v, preds.segment_id.to_numpy()),
           "n_breaks": int(n_breaks), "frac_unmatched": float((preds.pred_arc_id < 0).mean())}
    if path_arcs is not None:
        out["path_continuity"] = path_continuity(path_arcs, u, v)
    if labels is not None:
        out["top1"] = top1_from_preds(preds, labels)
    if route is not None:
        out.update(route_precision_recall(path_arcs if path_arcs is not None else preds.pred_arc_id.to_numpy(),
                                          route, lengths))
    return out
