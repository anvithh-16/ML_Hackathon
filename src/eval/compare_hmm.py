"""Classifier-only vs classifier + HMM/Viterbi on student_test (organiser report §5). Owner: Member 3.

python -m src.eval.compare_hmm --models xgb__default rforest__default --partition student_test
Always includes the distance-only HMM and nearest-candidate baseline (no ML model).
Needs: data/processed/<partition>/{candidates,labels}.parquet and results/scores/<model>__<variant>/<partition>.parquet
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common.config import load_config, load_hmm_config, path_of
from src.common.logging_utils import get_logger
from src.data.geo import project
from src.data.loader import load_route, load_trip, trip_dir
from src.eval.route_metrics import summarise
from src.hmm.decode import classifier_only, decode_trip
from src.hmm.graph import RoadGraph

log = get_logger("compare_hmm")


def _read(path):
    d = pd.read_parquet(path)
    d["trip_id"] = d.trip_id.astype(str)
    return d


def evaluate_trip(trip, cands, labels, route, scores, hcfg, cfg, graph):
    """Rows for classifier-only and HMM using the same scores (None = distance-only)."""
    u, v, L = graph.u, graph.v, graph.length
    clf = classifier_only(cands, scores, sigma_m=hcfg["sigma_m"])
    clf = trip.track[["obs_idx"]].merge(clf, on="obs_idx", how="left")
    clf["trip_id"] = trip.trip_id
    clf["pred_arc_id"] = clf.pred_arc_id.fillna(-1).astype(np.int64)
    clf["segment_id"] = 0
    res = decode_trip(trip, cands, scores, hcfg, cfg, graph)
    a = summarise(clf, None, labels, route, u, v, L)
    b = summarise(res.preds, res.path_arcs, labels, route, u, v, L, res.n_breaks)
    b["decode_s"] = res.timings["decode_s"]
    return a, b, clf, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help="score folders under results/scores, e.g. xgb__default")
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--maps", type=int, default=2, help="maps for best and worst trip of the first model")
    a = ap.parse_args()
    cfg, hcfg = load_config(), load_hmm_config()
    proc = path_of(cfg, "processed") / a.partition
    cands_all, labels_all = _read(proc / "candidates.parquet"), _read(proc / "labels.parquet")
    score_sets = {"distance-only": None}
    for m in a.models:
        score_sets[m] = _read(path_of(cfg, "results") / "scores" / m / f"{a.partition}.parquet")
    root = path_of(cfg, "data_root")
    rows, keep = [], {}
    for tid in sorted(cands_all.trip_id.unique()):
        d = trip_dir(root, a.partition, tid)
        trip = project(load_trip(d, a.partition))
        graph = RoadGraph(trip)
        route = load_route(d, len(trip.arcs))
        cands = cands_all[cands_all.trip_id == tid]
        labels = labels_all[labels_all.trip_id == tid]
        for name, sc in score_sets.items():
            s = None if sc is None else sc[sc.trip_id == tid]
            ca, hb, clf, res = evaluate_trip(trip, cands, labels, route, s, hcfg, cfg, graph)
            rows.append({"trip_id": tid, "scores": name, "method": "classifier-only", **ca})
            rows.append({"trip_id": tid, "scores": name, "method": "classifier+HMM", **hb})
            keep[(tid, name)] = (clf, res)
        log.info("done %s", tid)
    per_trip = pd.DataFrame(rows)
    out = path_of(cfg, "results") / "hmm"
    (out / "maps").mkdir(parents=True, exist_ok=True)
    per_trip.to_csv(out / "per_trip.csv", index=False)
    num = [c for c in per_trip.columns if c not in ("trip_id", "scores", "method")]
    summary = per_trip.groupby(["scores", "method"])[num].mean().reset_index()
    summary.to_csv(out / "clf_vs_hmm.csv", index=False)
    print(summary.round(4).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        (out / "figures").mkdir(exist_ok=True)
        piv = summary.pivot(index="scores", columns="method", values="top1")
        ax = piv.plot.bar(figsize=(8, 4.5), rot=20)
        ax.set_ylabel(f"Top-1 accuracy ({a.partition})")
        ax.set_title("Classifier-only vs classifier + HMM")
        ax.set_ylim(0, 1.0)
        ax.legend(loc="lower right", ncol=2, title="method")
        plt.tight_layout()
        plt.savefig(out / "figures" / "clf_vs_hmm_top1.png", dpi=150)
    except Exception as e:  # plotting must never break the experiment
        log.warning("plot failed: %s", e)

    if a.maps and a.models:
        from src.viz.folium_maps import trip_map
        name = a.models[0]
        hm = per_trip[(per_trip.scores == name) & (per_trip.method == "classifier+HMM")].sort_values("top1")
        for tid in [hm.trip_id.iloc[0], hm.trip_id.iloc[-1]][: a.maps]:
            d = trip_dir(root, a.partition, tid)
            trip = project(load_trip(d, a.partition))
            clf, res = keep[(tid, name)]
            path = [x for seg in res.path_arcs for x in (seg + [-1])]
            m = trip_map(trip, route_arc_ids=load_route(d), pred_arc_ids=path,
                         extra_pred={"Classifier-only (argmax)": clf.pred_arc_id.tolist()},
                         title=f"{a.partition} {tid} · {name}")
            m.save(str(out / "maps" / f"{tid}_{name}.html"))


if __name__ == "__main__":
    main()
