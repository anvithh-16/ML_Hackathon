"""Aggregate run records into the organiser comparison table + plots. Owner: Member 2.

python -m src.eval.compare_models
Reads the latest non-synthetic record per (model, variant) in results/experiments/, adds route
continuity of the classifier-only path on student_test (M3's route_metrics) and a non-ML
nearest-candidate baseline row. Every cell comes from a file - nothing is typed by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config, path_of, project_root
from src.common.logging_utils import get_logger
from src.data.loader import load_trip, trip_dir
from src.eval.experiment_log import load_experiments
from src.eval.metrics import evaluate_scores, top1_table
from src.eval.plots import comparison_plots, confusion_plots
from src.eval.route_metrics import route_continuity

log = get_logger("compare_models")


def _arcs_uv(partition: str, trip_ids, cfg):
    root = path_of(cfg, "data_root")
    out = {}
    for t in trip_ids:
        a = load_trip(trip_dir(root, partition, t), partition).arcs
        out[t] = (a.u.to_numpy(), a.v.to_numpy())
    return out


def classifier_only_continuity(feat: pd.DataFrame, scores: np.ndarray, uv: dict) -> float:
    top = top1_table(feat, scores).sort_values(["trip_id", "obs_idx"])
    vals = []
    for t, g in top.groupby("trip_id"):
        u, v = uv[t]
        vals.append(route_continuity(g.arc_id.to_numpy(), u, v))
    return float(np.nanmean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="student_test")
    a = ap.parse_args()
    cfg = load_config()
    exps = load_experiments(latest_only=True)
    if exps.empty:
        raise SystemExit("no non-synthetic run records in results/experiments/ - run src.eval.cv_runner first")

    feat = pd.read_parquet(path_of(cfg, "processed") / a.partition / "features.parquet",
                           columns=["trip_id", "obs_idx", "arc_id", "dist_m", "y", "labelled"])
    feat["trip_id"] = feat.trip_id.astype(str)
    uv = _arcs_uv(a.partition, sorted(feat.trip_id.unique()), cfg)
    lab = feat[feat.labelled].reset_index(drop=True)

    rows, conf = [], {}
    # non-ML baseline: nearest candidate
    base = evaluate_scores(lab, -lab.dist_m.to_numpy())
    rows.append({"Model": "Nearest-candidate baseline (no ML)", "Algorithm family": "rule", "Key configuration": "argmin distance",
                 "CV Macro F1": "N/A (no training)", "Top-1 Accuracy": base["top1"],
                 "Route continuity": classifier_only_continuity(lab, -lab.dist_m.to_numpy(), uv),
                 "Student-Test Macro F1": "N/A (no probabilities)", "Log Loss": "N/A (no probabilities)",
                 "Inference time (ms / 1k rows)": "N/A", "Precision": "N/A", "Recall": "N/A",
                 "Training time (s)": "N/A", "Candidate miss rate": base["cand_miss_rate"]})
    sdir = path_of(cfg, "results") / "scores"
    for rec in exps.to_dict("records"):
        name = f"{rec['model']}__{rec['variant']}"
        cont = np.nan
        sf = sdir / name / f"{a.partition}.parquet"
        if sf.exists():
            s = pd.read_parquet(sf)
            s["trip_id"] = s.trip_id.astype(str)
            m = lab.merge(s, on=["trip_id", "obs_idx", "arc_id"], how="left")
            cont = classifier_only_continuity(m, m.score.fillna(0).to_numpy(), uv)
        params = json.loads(open(rec["_file"]).read()).get("params", {})   # exact values incl. None
        cvm, cvs = rec.get("cv.macro_f1_mean"), rec.get("cv.macro_f1_std")
        rows.append({"Model": name, "Algorithm family": rec.get("family"),
                     "Key configuration": json.dumps(params)[:120],
                     "CV Macro F1": f"{cvm:.4f} ± {cvs:.4f}" if pd.notna(cvm) else "not run",
                     "Top-1 Accuracy": rec.get("test.top1"), "Route continuity": cont,
                     "Student-Test Macro F1": rec.get("test.macro_f1"), "Log Loss": rec.get("test.logloss"),
                     "Inference time (ms / 1k rows)": rec.get("infer_ms_per_1k"),
                     "Precision": rec.get("test.precision"), "Recall": rec.get("test.recall"),
                     "Training time (s)": rec.get("train_time_s"), "Candidate miss rate": rec.get("test.cand_miss_rate"),
                     "CV Top-1": rec.get("cv.top1_mean"), "record": Path(rec["_file"]).relative_to(project_root()).as_posix() if rec.get("_file") else None})
        if isinstance(rec.get("test.confusion"), list):
            conf[name] = np.array(rec["test.confusion"])
    table = pd.DataFrame(rows)
    out = path_of(cfg, "results")
    table.to_csv(out / "model_comparison.csv", index=False)
    print(table.drop(columns=["record"], errors="ignore").to_string(index=False))
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    comparison_plots(exps, figs)
    confusion_plots(conf, figs)
    print(f"\nwrote {out / 'model_comparison.csv'} and figures in {figs}")


if __name__ == "__main__":
    main()
