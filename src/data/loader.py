"""Read raw PathSnap trip files. Owner: Member 1.

Folder layout:  <root>/<partition>/<trip_id>/<trip_id>.{track,nodes,arcs,route}
The route is NEVER read by load_trip(); only load_route() reads it (labels / server-side scoring).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config, path_of
from src.common.schema import Trip, check_trip_id


class TripFormatError(ValueError):
    """A trip folder or file does not match the expected raw format."""


def _is_trip_dir(p: Path) -> bool:
    return p.is_dir() and len(p.name) == 8 and p.name.isdigit()


def list_trips(root: str | Path, partition: str | None = None) -> list[str]:
    """Sorted 8-digit trip ids. partition=None means root itself holds trip folders."""
    base = Path(root) / partition if partition else Path(root)
    if not base.exists():
        raise FileNotFoundError(f"Trip folder not found: {base}")
    return sorted(p.name for p in base.iterdir() if _is_trip_dir(p))


def trip_dir(root: str | Path, partition: str | None, trip_id: str) -> Path:
    trip_id = check_trip_id(trip_id)
    return (Path(root) / partition / trip_id) if partition else (Path(root) / trip_id)


def _file(path: Path, ext: str) -> Path:
    f = path / f"{path.name}.{ext}"
    if not f.exists():
        raise TripFormatError(f"Missing file {f}")
    return f


def _read(f: Path, names: list[str], dtype) -> pd.DataFrame:
    try:
        df = pd.read_csv(f, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")
    except ValueError as e:  # non-numeric content, wrong column count ...
        raise TripFormatError(f"{f}: cannot parse as {len(names)} numeric columns ({e})") from e
    if df.shape[1] != len(names):
        raise TripFormatError(f"{f}: expected {len(names)} columns, got {df.shape[1]}")
    if df.isna().any().any():
        raise TripFormatError(f"{f}: contains missing / non-numeric values or wrong column count")
    return df


def load_trip(path: str | Path, partition: str | None = None) -> Trip:
    """Load track, nodes and arcs of one trip (unprojected: x/y are NaN until geo.project)."""
    path = Path(path)
    if not path.is_dir():
        raise TripFormatError(f"Not a trip folder: {path}")
    tid = check_trip_id(path.name)

    track = _read(_file(path, "track"), ["lon", "lat", "t"], float)
    nodes = _read(_file(path, "nodes"), ["lon", "lat"], float)
    arcs = _read(_file(path, "arcs"), ["u", "v"], np.int64)

    if len(track) == 0 or len(nodes) == 0 or len(arcs) == 0:
        raise TripFormatError(f"{tid}: empty track/nodes/arcs")
    for name, df in (("track", track), ("nodes", nodes)):
        if not df["lon"].between(-180, 180).all() or not df["lat"].between(-90, 90).all():
            raise TripFormatError(f"{tid}.{name}: coordinates out of range (lon/lat swapped?)")
    meta = {}
    if (np.diff(track["t"].to_numpy()) <= 0).any():
        # Not seen in the 90 supplied trips, but evaluator trips must never crash the API:
        # sort by time, drop duplicated timestamps, and record that we did it.
        n0 = len(track)
        track = track.sort_values("t", kind="stable").drop_duplicates("t").reset_index(drop=True)
        meta["time_repaired"] = f"sorted/deduplicated timestamps ({n0} -> {len(track)} rows)"
    if arcs[["u", "v"]].to_numpy().min() < 0 or arcs[["u", "v"]].to_numpy().max() >= len(nodes):
        raise TripFormatError(f"{tid}.arcs: node id outside 0..{len(nodes) - 1}")

    track.insert(0, "obs_idx", np.arange(len(track), dtype=np.int64))
    track["x"] = np.nan
    track["y"] = np.nan
    nodes.insert(0, "node_id", np.arange(len(nodes), dtype=np.int64))
    nodes["x"] = np.nan
    nodes["y"] = np.nan
    arcs.insert(0, "arc_id", np.arange(len(arcs), dtype=np.int64))
    arcs["length_m"] = np.nan
    arcs["bearing_deg"] = np.nan
    arcs["reverse_arc_id"] = np.int64(-1)
    return Trip(trip_id=tid, partition=partition, track=track, nodes=nodes, arcs=arcs, epsg=None, meta=meta)


def load_route(path: str | Path, n_arcs: int | None = None) -> np.ndarray:
    """Reference route = chronological arc ids. Only for labels and server-side scoring."""
    f = _file(Path(path), "route")
    route = _read(f, ["arc_id"], np.int64)["arc_id"].to_numpy()
    if n_arcs is not None and len(route) and (route.min() < 0 or route.max() >= n_arcs):
        raise TripFormatError(f"{f}: arc id outside 0..{n_arcs - 1}")
    return route


def has_route(path: str | Path) -> bool:
    p = Path(path)
    return (p / f"{p.name}.route").exists()


def main() -> None:
    ap = argparse.ArgumentParser(description="Load one trip and print a summary")
    ap.add_argument("--trip", required=True)
    ap.add_argument("--partition", default="train")
    a = ap.parse_args()
    cfg = load_config()
    d = trip_dir(path_of(cfg, "data_root"), a.partition, a.trip)
    t = load_trip(d, a.partition)
    r = load_route(d, len(t.arcs)) if has_route(d) else np.array([])
    print(f"trip {t.trip_id}: {len(t.track)} obs, {len(t.nodes)} nodes, {len(t.arcs)} arcs, {len(r)} route entries")


if __name__ == "__main__":
    main()
