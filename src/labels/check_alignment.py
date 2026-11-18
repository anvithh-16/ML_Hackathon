"""Silent-error guard for labels: per-trip alignment summary + Folium overlays. Owner: Member 1.

python -m src.labels.check_alignment --trips 00000001 00000010 00000041 00000054 00000075
Requires data/processed/<partition>/labels.parquet (built by build_features or align_route).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common.config import load_config, path_of
from src.data.loader import load_route, load_trip, trip_dir
from src.data.geo import project
from src.viz.folium_maps import trip_map


def summary(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (part, tid), g in labels.groupby(["partition", "trip_id"]):
        lab = g.label_arc_id >= 0
        rp = g.route_pos[g.route_pos >= 0].to_numpy()
        rows.append({"partition": part, "trip_id": tid, "n_obs": len(g), "frac_labelled": lab.mean(),
                     "align_dist_median_m": g.align_dist_m.median(),
                     "align_dist_p95_m": g.align_dist_m.quantile(0.95),
                     "route_pos_monotonic": bool((np.diff(rp) >= 0).all()) if len(rp) > 1 else True,
                     "route_positions_covered": int(np.unique(rp).size)})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="*", default=["00000001", "00000010", "00000041", "00000054", "00000075"])
    ap.add_argument("--partition", default="train")
    a = ap.parse_args()
    out = path_of(cfg, "results") / "labels"
    (out / "maps").mkdir(parents=True, exist_ok=True)
    frames = []
    for part in cfg["partitions"]:
        f = path_of(cfg, "processed") / part / "labels.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            d["trip_id"] = d.trip_id.astype(str)
            d["partition"] = part
            frames.append(d)
    if not frames:
        raise SystemExit("No labels.parquet found - run src.features.build_features first")
    labels = pd.concat(frames, ignore_index=True)
    s = summary(labels)
    s.to_csv(out / "alignment_summary.csv", index=False)
    print(s.describe().round(3).to_string())
    bad = s[(s.frac_labelled < 0.95) | (~s.route_pos_monotonic)]
    print("\nTrips needing attention:\n", bad.to_string(index=False) if len(bad) else "none")

    root = path_of(cfg, "data_root")
    for tid in a.trips:
        d = trip_dir(root, a.partition, tid)
        trip = project(load_trip(d, a.partition))
        lab = labels[(labels.trip_id == tid) & (labels.partition == a.partition)]
        m = trip_map(trip, route_arc_ids=load_route(d), labels=lab, title=f"{a.partition} {tid}: alignment check")
        m.save(str(out / "maps" / f"align_{tid}.html"))
        print("map:", out / "maps" / f"align_{tid}.html")


if __name__ == "__main__":
    main()
