"""Static overlays for the trips flagged by the alignment check (report §2.5). Owner: Member 1.

Left panel: reference route (green), labelled GPS points (blue), unlabelled (-1) points (red).
Right panel: aligned route position against observation index (a backward step = a restart of the alignment).

Run: python -m eda.flagged_trips
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.config import load_config, path_of
from src.data.geo import arc_endpoints, project
from src.data.loader import load_route, load_trip, trip_dir

FLAGGED = [("train", "00000002"), ("train", "00000043"), ("train", "00000066"), ("train", "00000081"),
           ("student_test", "00000056"), ("train", "00000001")]


def main() -> None:
    cfg = load_config()
    root = path_of(cfg, "data_root")
    out = path_of(cfg, "results") / "labels" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for part, tid in FLAGGED:
        d = trip_dir(root, part, tid)
        trip = project(load_trip(d, part))
        route = load_route(d, len(trip.arcs))
        lab = pd.read_parquet(path_of(cfg, "processed") / part / "labels.parquet")
        lab["trip_id"] = lab.trip_id.astype(str)
        lab = lab[lab.trip_id == tid].sort_values("obs_idx")
        A, B = arc_endpoints(trip)
        tr = trip.track
        un = (lab.label_arc_id.to_numpy() < 0)
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
        for a in route:
            ax[0].plot([A[a, 0], B[a, 0]], [A[a, 1], B[a, 1]], color="#2e7d32", lw=2.2, zorder=1)
        ax[0].scatter(tr.x[~un], tr.y[~un], s=3, c="#1565c0", zorder=2, label=f"labelled ({(~un).sum()})")
        ax[0].scatter(tr.x[un], tr.y[un], s=6, c="#c62828", zorder=3, label=f"label -1 ({un.sum()})")
        ax[0].set_aspect("equal", "datalim")
        ax[0].set_title(f"{part} {tid}: GPS vs reference route")
        ax[0].set_xlabel("x (m, UTM)")
        ax[0].set_ylabel("y (m, UTM)")
        ax[0].legend(loc="best", fontsize=8, markerscale=3)
        rp = lab.route_pos.to_numpy().astype(float)
        rp[un] = np.nan
        back = int((np.diff(rp[~np.isnan(rp)]) < 0).sum())
        ax[1].plot(lab.obs_idx, rp, ".", ms=2, color="#1565c0")
        ax[1].set_title(f"aligned route position (backward steps: {back})")
        ax[1].set_xlabel("GPS observation index")
        ax[1].set_ylabel("route position")
        fig.tight_layout()
        fig.savefig(out / f"align_{part}_{tid}.png", dpi=110)
        plt.close(fig)
        print(f"{part} {tid}: labelled {(~un).mean():.3f}, backward steps {back} -> {out / f'align_{part}_{tid}.png'}")


if __name__ == "__main__":
    main()
