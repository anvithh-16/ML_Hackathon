"""Dataset inventory, integrity checks, anomalies and network connectivity. Owner: Member 1.

python -m src.data.validate                      # both partitions -> results/eda/*.csv
python -m src.data.validate --root tests/fixtures/mini_trip --flat
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.common.config import load_config, path_of, resolve
from src.common.logging_utils import get_logger
from src.data.geo import arc_endpoints, point_segment, project
from src.data.loader import TripFormatError, has_route, list_trips, load_route, load_trip

log = get_logger("validate")


def connectivity(trip) -> dict:
    n = len(trip.nodes)
    u, v = trip.arcs["u"].to_numpy(), trip.arcs["v"].to_numpy()
    g = coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n)).tocsr()
    n_comp, lab = connected_components(g, directed=True, connection="weak")
    sizes = np.bincount(lab)
    deg = np.bincount(u, minlength=n) + np.bincount(v, minlength=n)
    return {"n_components": int(n_comp), "largest_component_share": float(sizes.max() / n),
            "isolated_nodes": int((deg == 0).sum())}


def trip_row(path: Path, partition: str | None) -> dict:
    row = {"partition": partition or "", "trip_id": path.name}
    files = {e: (path / f"{path.name}.{e}") for e in ("track", "nodes", "arcs", "route")}
    for e, f in files.items():
        row[f"has_{e}"] = f.exists()
        row[f"{e}_bytes"] = f.stat().st_size if f.exists() else 0
    try:
        trip = project(load_trip(path, partition))
    except TripFormatError as e:
        row["error"] = str(e)
        return row
    tr, ar = trip.track, trip.arcs
    dt = np.diff(tr["t"].to_numpy())
    row.update({
        "n_obs": len(tr), "n_nodes": len(trip.nodes), "n_arcs": len(ar),
        "epsg": trip.epsg, "lon_mean": tr.lon.mean(), "lat_mean": tr.lat.mean(),
        "extent_x_km": (tr.x.max() - tr.x.min()) / 1000, "extent_y_km": (tr.y.max() - tr.y.min()) / 1000,
        "duration_s": float(tr.t.iloc[-1] - tr.t.iloc[0]),
        "dt_median_s": float(np.median(dt)) if len(dt) else np.nan,
        "dt_max_s": float(dt.max()) if len(dt) else np.nan,
        "frac_dt_1s": float(np.mean(dt == 1)) if len(dt) else np.nan,
        "n_gaps_gt_30s": int((dt > 30).sum()),
        "missing_values": 0, "time_repaired": trip.meta.get("time_repaired", ""),
        "self_loops": int((ar.u == ar.v).sum()),
        "duplicate_node_coords": int(len(trip.nodes) - len(trip.nodes[["lon", "lat"]].drop_duplicates())),
        "reverse_arc_frac": float((ar.reverse_arc_id >= 0).mean()),
        "arc_len_median_m": float(ar.length_m.median()),
    })
    row.update(connectivity(trip))
    if has_route(path):
        r = load_route(path, len(ar))
        u, v = ar.u.to_numpy(), ar.v.to_numpy()
        row.update({"n_route": len(r), "route_unique_arcs": len(np.unique(r)),
                    "route_continuity": float(np.mean(v[r[:-1]] == u[r[1:]])) if len(r) > 1 else np.nan,
                    "obs_per_route_arc": len(tr) / max(len(r), 1)})
        A, B = arc_endpoints(trip)
        P = tr[["x", "y"]].to_numpy()
        sub = np.arange(0, len(P), max(1, len(P) // 500))
        ru = np.unique(r)
        d = np.array([point_segment(np.repeat(P[i:i + 1], len(ru), 0), A[ru], B[ru])[0].min() for i in sub])
        row.update({"gps_to_route_median_m": float(np.median(d)), "gps_to_route_p95_m": float(np.percentile(d, 95)),
                    "frac_gps_gt50m_from_route": float(np.mean(d > 50))})
    return row


def inventory(root: Path, partitions: list[str] | None) -> pd.DataFrame:
    rows = []
    for part in partitions or [None]:
        for tid in list_trips(root, part):
            path = (root / part / tid) if part else (root / tid)
            rows.append(trip_row(path, part))
            log.info("inventoried %s %s", part, tid)
    return pd.DataFrame(rows)


def anomalies(inv: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rules = [
        ("missing_file", lambda r: not (r.has_track and r.has_nodes and r.has_arcs), "trip cannot be used"),
        ("parse_error", lambda r: isinstance(r.get("error"), str) and r.get("error") != "", "trip cannot be used"),
        ("gap_gt_split", lambda r: r.get("dt_max_s", 0) > cfg["gap_split_s"], "HMM splits into segments; dt capped in features"),
        ("low_1hz_fraction", lambda r: r.get("frac_dt_1s", 1) < 0.9, "irregular sampling; kept"),
        ("route_discontinuity", lambda r: r.get("route_continuity", 1) < 1.0, "reference route has jumps; labels by alignment"),
        ("gps_far_from_route", lambda r: r.get("frac_gps_gt50m_from_route", 0) > 0.10, "reference route unreliable; excluded from training"),
        ("self_loops", lambda r: r.get("self_loops", 0) > 0, "kept; no reverse twin assigned"),
        ("time_repaired", lambda r: bool(r.get("time_repaired", "")), "timestamps sorted/deduplicated"),
    ]
    out = []
    for _, r in inv.iterrows():
        for name, fn, decision in rules:
            try:
                hit = fn(r)
            except (KeyError, TypeError):
                hit = False
            if hit:
                out.append({"partition": r.partition, "trip_id": r.trip_id, "rule": name, "decision": decision})
    return pd.DataFrame(out, columns=["partition", "trip_id", "rule", "decision"])


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Dataset inventory and integrity checks")
    ap.add_argument("--root", default=str(path_of(cfg, "data_root")))
    ap.add_argument("--flat", action="store_true", help="root contains trip folders directly")
    ap.add_argument("--out", default=str(path_of(cfg, "results") / "eda"))
    a = ap.parse_args()
    root, out = resolve(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inv = inventory(root, None if a.flat else cfg["partitions"])
    an = anomalies(inv, cfg)
    conn_cols = ["partition", "trip_id", "n_nodes", "n_arcs", "n_components", "largest_component_share", "isolated_nodes"]
    inv.to_csv(out / "trip_inventory.csv", index=False)
    an.to_csv(out / "anomalies.csv", index=False)
    inv[[c for c in conn_cols if c in inv]].to_csv(out / "network_connectivity.csv", index=False)
    print(inv.groupby("partition").agg(trips=("trip_id", "size"), obs=("n_obs", "sum")).to_string())
    print(f"anomalies: {len(an)} rows -> {out / 'anomalies.csv'}")


if __name__ == "__main__":
    main()
