"""Overfitting evidence: training-fold vs validation-fold scores, same folds/subsample as the CV runner.

python -m src.eval.train_vs_val --model dtree                 # default params, per fold
python -m src.eval.train_vs_val --model dtree --depths 3 6 12 20 30 0   # depth sweep (0 = unlimited)

Writes results/eval/<model>_train_vs_val.csv (and <model>_depth_sweep.csv). Training rows are the
fold's training trips; validation rows are the held-out trips. student_test is not used here.
Owner: Member 2.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common.config import load_config, load_model_config, path_of
from src.common.logging_utils import get_logger
from src.common.schema import FEATURE_COLUMNS
from src.eval.cv_runner import load_folds, load_partition_features
from src.eval.metrics import evaluate_scores
from src.models.registry import get_model_class

log = get_logger("train_vs_val")


def _fold_scores(cls, params: dict, df: pd.DataFrame, fold_of: pd.Series, seed: int) -> list[dict]:
    rows = []
    for k in sorted(fold_of.unique()):
        tr, va = df[fold_of != k], df[fold_of == k]
        assert set(tr.trip_id).isdisjoint(set(va.trip_id)), "train/validation trips must be disjoint"
        m = cls(params=params, seed=seed)
        m.fit(tr[FEATURE_COLUMNS], tr["y"])
        a = evaluate_scores(tr.reset_index(drop=True), m.predict_proba(tr[FEATURE_COLUMNS]))
        b = evaluate_scores(va.reset_index(drop=True), m.predict_proba(va[FEATURE_COLUMNS]))
        rows.append({"fold": int(k), "n_train_rows": len(tr), "n_val_rows": len(va),
                     "train_top1": a["top1"], "val_top1": b["top1"], "gap_top1": a["top1"] - b["top1"],
                     "train_macro_f1": a["macro_f1"], "val_macro_f1": b["macro_f1"],
                     "gap_macro_f1": a["macro_f1"] - b["macro_f1"],
                     "train_logloss": a["logloss"], "val_logloss": b["logloss"]})
        log.info("fold %s: top1 train %.4f val %.4f", k, a["top1"], b["top1"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dtree")
    ap.add_argument("--depths", nargs="*", type=int, help="optional max_depth sweep (dtree only); 0 = unlimited")
    a = ap.parse_args()
    cfg = load_config()
    mc = load_model_config(a.model)
    cls = get_model_class(a.model)
    df = load_partition_features("train", cfg, cfg["training"]["obs_stride"]).reset_index(drop=True)   # same subsample as the CV runner
    fold_of = df["trip_id"].astype(str).map(load_folds()["folds"])
    if fold_of.isna().any():
        raise SystemExit("some training trips have no fold in configs/folds.json")
    out = path_of(cfg, "results") / "eval"
    out.mkdir(parents=True, exist_ok=True)

    rows = _fold_scores(cls, mc["params"], df, fold_of, cfg["seed"])
    t = pd.DataFrame(rows)
    mean = t.drop(columns="fold").mean().to_dict()
    t = pd.concat([t, pd.DataFrame([{"fold": "mean", **mean}])], ignore_index=True)
    t.to_csv(out / f"{a.model}_train_vs_val.csv", index=False)
    print(t.round(4).to_string(index=False))

    if a.depths:
        sweep = []
        for d in a.depths:
            p = dict(mc["params"], max_depth=None if d == 0 else d)
            r = pd.DataFrame(_fold_scores(cls, p, df, fold_of, cfg["seed"]))
            sweep.append({"max_depth": "unlimited" if d == 0 else d,
                          "train_top1": r.train_top1.mean(), "val_top1": r.val_top1.mean(),
                          "gap_top1": r.gap_top1.mean(), "val_top1_std": r.val_top1.std(ddof=0),
                          "train_macro_f1": r.train_macro_f1.mean(), "val_macro_f1": r.val_macro_f1.mean(),
                          "train_logloss": r.train_logloss.mean(), "val_logloss": r.val_logloss.mean()})
        s = pd.DataFrame(sweep)
        s.to_csv(out / f"{a.model}_depth_sweep.csv", index=False)
        print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
