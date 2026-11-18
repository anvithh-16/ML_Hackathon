"""Metric tests on hand-computed inputs. Owner: Member 2."""
import numpy as np
import pandas as pd

from src.eval.metrics import candidate_miss_rate, evaluate_scores, safe_log_loss, top1_accuracy


def table():
    # obs 0: correct arc 11 scored highest; obs 1: tie -> smaller dist wins (correct);
    # obs 2: correct arc not among candidates -> wrong
    return pd.DataFrame({
        "trip_id": ["00000001"] * 7, "obs_idx": [0, 0, 1, 1, 1, 2, 2],
        "arc_id": [10, 11, 20, 21, 22, 30, 31], "dist_m": [5, 3, 9, 2, 4, 1, 2],
        "y": [0, 1, 0, 1, 0, 0, 0], "labelled": True})


def test_top1():
    df = table()
    s = np.array([0.2, 0.9, 0.5, 0.5, 0.1, 0.7, 0.3])
    assert abs(top1_accuracy(df, s) - 2 / 3) < 1e-12


def test_miss_rate():
    assert abs(candidate_miss_rate(table()) - 1 / 3) < 1e-12


def test_logloss_finite_with_hard_probs():
    assert np.isfinite(safe_log_loss([0, 1, 1], [0.0, 1.0, 0.0]))


def test_evaluate_scores_keys():
    m = evaluate_scores(table(), np.array([0.2, 0.9, 0.5, 0.5, 0.1, 0.7, 0.3]))
    for k in ("top1", "macro_f1", "logloss", "precision", "recall", "cand_miss_rate", "confusion"):
        assert k in m
