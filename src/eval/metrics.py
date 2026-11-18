"""The ONLY implementation of classification metrics in PathSnap. Owner: Member 2.

Inputs are candidate-level tables with columns trip_id, obs_idx, arc_id, dist_m, y, labelled
(i.e. rows of features.parquet) plus a score array aligned with those rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score

from src.common.schema import PROBA_CLIP


def top1_table(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """One row per observation: the top-scoring candidate (ties -> smaller dist_m, then arc_id)."""
    d = df[["trip_id", "obs_idx", "arc_id", "dist_m", "y"]].copy()
    d["score"] = np.asarray(scores, dtype=float)
    d = d.sort_values(["trip_id", "obs_idx", "score", "dist_m", "arc_id"],
                      ascending=[True, True, False, True, True], kind="mergesort")
    return d.drop_duplicates(["trip_id", "obs_idx"], keep="first")


def top1_accuracy(df: pd.DataFrame, scores: np.ndarray) -> float:
    """Share of labelled observations whose top candidate is the label arc.

    Observations whose label arc is not among the candidates count as wrong (their y are all 0).
    df must contain only labelled rows.
    """
    if len(df) == 0:
        return float("nan")
    return float(top1_table(df, scores)["y"].mean())


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0))


def safe_log_loss(y_true, p) -> float:
    p = np.clip(np.asarray(p, float), PROBA_CLIP, 1 - PROBA_CLIP)
    return float(log_loss(y_true, p, labels=[0, 1]))


def precision_recall(y_true, y_pred) -> dict:
    return {"precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0))}


def candidate_miss_rate(df: pd.DataFrame) -> float:
    """Share of labelled observations whose label arc is absent from the candidate set."""
    if len(df) == 0:
        return float("nan")
    g = df.groupby(["trip_id", "obs_idx"])["y"].max()
    return float(1 - g.mean())


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def evaluate_scores(df: pd.DataFrame, scores: np.ndarray, threshold: float = 0.5) -> dict:
    """All candidate-level metrics on labelled rows. df rows and scores must be aligned."""
    scores = np.asarray(scores, float)
    mask = df["labelled"].to_numpy(bool) if "labelled" in df else np.ones(len(df), bool)
    d, s = df[mask], scores[mask]
    y = d["y"].to_numpy().astype(int)
    yp = (s >= threshold).astype(int)
    out = {"top1": top1_accuracy(d, s), "macro_f1": macro_f1(y, yp), "logloss": safe_log_loss(y, s),
           "cand_miss_rate": candidate_miss_rate(d), "n_obs": int(d.groupby(["trip_id", "obs_idx"]).ngroups),
           "n_rows": int(len(d))}
    out.update(precision_recall(y, yp))
    out["confusion"] = confusion(y, yp).tolist()
    return out
