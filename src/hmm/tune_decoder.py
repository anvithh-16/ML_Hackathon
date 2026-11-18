"""Grid search of decoder parameters on TRAINING out-of-fold scores. Owner: Member 3.

python -m src.hmm.tune_decoder --scores xgb__default --n-trips 20
Uses results/scores/<model>__<variant>/oof_train.parquet (strided observations) so the
student_test partition stays untouched for the final comparison. Writes
results/hmm/decoder_sweep.csv and results/hmm/best_decoder.yaml (copy values into configs/hmm.yaml).
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
import yaml

from src.common.config import load_config, load_hmm_config, path_of
from src.common.logging_utils import get_logger
from src.data.geo import project
from src.data.loader import load_trip, trip_dir
from src.eval.route_metrics import route_continuity, top1_from_preds
from src.hmm.decode import decode_trip
from src.hmm.graph import RoadGraph

log = get_logger("tune_decoder")
GRID = {"beta_m": [2, 5, 10, 20, 40], "uturn_penalty": [0.0, 2.0, 5.0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="e.g. xgb__default (folder in results/scores)")
    ap.add_argument("--n-trips", type=int, default=20)
    a = ap.parse_args()
    cfg, base = load_config(), load_hmm_config()
    sc = pd.read_parquet(path_of(cfg, "results") / "scores" / a.scores / "oof_train.parquet")
    sc["trip_id"] = sc.trip_id.astype(str)
    proc = path_of(cfg, "processed") / "train"
    cands_all = pd.read_parquet(proc / "candidates.parquet")
    cands_all["trip_id"] = cands_all.trip_id.astype(str)
    labels_all = pd.read_parquet(proc / "labels.parquet")
    labels_all["trip_id"] = labels_all.trip_id.astype(str)
    trips = sorted(sc.trip_id.unique())
    rng = np.random.default_rng(cfg["seed"])
    trips = sorted(rng.choice(trips, size=min(a.n_trips, len(trips)), replace=False))
    root = path_of(cfg, "data_root")
    prepared = []
    for tid in trips:
        trip = project(load_trip(trip_dir(root, "train", tid), "train"))
        s = sc[sc.trip_id == tid]
        prepared.append((trip, RoadGraph(trip), cands_all[cands_all.trip_id == tid], s,
                         labels_all[labels_all.trip_id == tid], np.sort(s.obs_idx.unique())))
    rows = []
    for beta, ut in itertools.product(GRID["beta_m"], GRID["uturn_penalty"]):
        h = dict(base, beta_m=beta, uturn_penalty=ut)
        t1, cont = [], []
        for trip, g, c, s, lab, subset in prepared:
            res = decode_trip(trip, c, s, h, cfg, g, obs_subset=subset)
            t1.append(top1_from_preds(res.preds, lab[lab.obs_idx.isin(subset)]))
            cont.append(route_continuity(res.preds.pred_arc_id.to_numpy(), g.u, g.v, res.preds.segment_id.to_numpy()))
        rows.append({"beta_m": beta, "uturn_penalty": ut, "hmm_top1_mean": float(np.nanmean(t1)),
                     "route_continuity_mean": float(np.nanmean(cont)), "n_trips": len(prepared)})
        log.info("beta=%s uturn=%s -> top1 %.4f", beta, ut, rows[-1]["hmm_top1_mean"])
    df = pd.DataFrame(rows).sort_values(["hmm_top1_mean", "route_continuity_mean"], ascending=False)
    out = path_of(cfg, "results") / "hmm"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "decoder_sweep.csv", index=False)
    best = dict(base, beta_m=float(df.iloc[0].beta_m), uturn_penalty=float(df.iloc[0].uturn_penalty))
    (out / "best_decoder.yaml").write_text(yaml.safe_dump(best, sort_keys=False))
    print(df.round(4).to_string(index=False))
    print("best ->", out / "best_decoder.yaml", "(copy values into configs/hmm.yaml)")


if __name__ == "__main__":
    main()
