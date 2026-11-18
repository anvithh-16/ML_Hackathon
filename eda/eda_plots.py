"""EDA figures for report §2. Owner: Member 1.

python -m eda.eda_plots
Needs results/eda/trip_inventory.csv (src.data.validate) and data/processed/* (build_features).
Every figure title names the partition it uses.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.config import load_config, path_of  # noqa: E402
from src.data.geo import project  # noqa: E402
from src.data.loader import load_trip, trip_dir  # noqa: E402


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    print("wrote", out / name)


def main() -> None:
    cfg = load_config()
    res = path_of(cfg, "results")
    out = res / "eda" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(res / "eda" / "trip_inventory.csv", dtype={"trip_id": str})

    fig, ax = plt.subplots(figsize=(7, 4))
    for part, g in inv.groupby("partition"):
        ax.hist(g.n_obs, bins=30, alpha=0.6, label=f"{part} ({len(g)} trips)")
    ax.set_xscale("log")
    ax.set_xlabel("GPS observations per trip (log)")
    ax.set_ylabel("trips")
    ax.set_title("Observations per trip (train + student_test)")
    ax.legend()
    save(fig, out, "obs_per_trip.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(inv.n_obs, inv.n_route, c=(inv.partition == "train"), cmap="coolwarm", s=18)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("GPS observations")
    ax.set_ylabel("reference route entries")
    ax.set_title("GPS rows vs route entries — they are NOT 1:1 (train + student_test)")
    save(fig, out, "obs_vs_route_len.png")

    root = path_of(cfg, "data_root")
    dts = []
    for r in inv.itertuples():
        t = pd.read_csv(root / r.partition / r.trip_id / f"{r.trip_id}.track", sep=r"\s+", header=None)[2].to_numpy()
        dts.append(np.diff(t))
    dt = np.concatenate(dts)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(dt, 0, 60), bins=60)
    ax.set_yscale("log")
    ax.set_xlabel("seconds between consecutive GPS points (clipped at 60)")
    ax.set_ylabel("count (log)")
    ax.set_title(f"Sampling interval (train + student_test); {np.mean(dt == 1):.1%} exactly 1 s")
    save(fig, out, "sampling_interval.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(inv.n_nodes, inv.n_arcs, s=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nodes")
    ax.set_ylabel("arcs")
    ax.set_title("Road-network size per trip (train + student_test)")
    save(fig, out, "network_size.png")

    lab = pd.read_parquet(path_of(cfg, "processed") / "train" / "labels.parquet")
    lab = lab[~lab.trip_id.astype(str).isin(cfg["exclude_trips_train"])]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lab.align_dist_m.dropna().clip(upper=100), bins=100)
    ax.set_xlabel("distance from GPS point to its aligned route arc (m, clipped at 100)")
    ax.set_ylabel("observations")
    ax.set_title("GPS noise relative to the reference route (train, excluded trips removed)")
    save(fig, out, "gps_to_route_distance.png")

    c = pd.read_parquet(path_of(cfg, "processed") / "train" / "candidates.parquet", columns=["trip_id", "obs_idx"])
    n = c.groupby(["trip_id", "obs_idx"]).size()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(n, bins=range(0, 32))
    ax.set_xlabel("candidate arcs per observation")
    ax.set_ylabel("observations")
    ax.set_title(f"Candidates per observation (train, radius {cfg['candidates']['radius_m']} m)")
    save(fig, out, "candidates_per_obs.png")

    for part, tid in (("train", "00000001"), ("student_test", "00000000")):
        trip = project(load_trip(trip_dir(root, part, tid), part))
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.plot(trip.track.x - trip.track.x.min(), trip.track.y - trip.track.y.min(), ".", ms=2)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m, UTM)")
        ax.set_ylabel("y (m, UTM)")
        ax.set_title(f"GPS trajectory {part} {tid} (EPSG {trip.epsg})")
        save(fig, out, f"trajectory_{part}_{tid}.png")


if __name__ == "__main__":
    main()
