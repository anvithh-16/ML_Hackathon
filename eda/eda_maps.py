"""Folium EDA maps (report §2 and §6). Owner: Member 1.

python -m eda.eda_maps --trips train:00000001 train:00000041 train:00000090 student_test:00000000
"""
from __future__ import annotations

import argparse

from src.common.config import load_config, path_of
from src.data.geo import project
from src.data.loader import load_route, load_trip, trip_dir
from src.viz.folium_maps import trip_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="*",
                    default=["train:00000001", "train:00000041", "train:00000090", "student_test:00000000"])
    a = ap.parse_args()
    cfg = load_config()
    root = path_of(cfg, "data_root")
    out = path_of(cfg, "results") / "eda" / "maps"
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.trips:
        part, tid = spec.split(":")
        d = trip_dir(root, part, tid)
        trip = project(load_trip(d, part))
        m = trip_map(trip, route_arc_ids=load_route(d), title=f"{part} {tid}: GPS, network, reference route")
        f = out / f"{part}_{tid}.html"
        m.save(str(f))
        print("wrote", f)


if __name__ == "__main__":
    main()
