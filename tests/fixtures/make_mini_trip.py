"""Generate the MOCK mini trip 00000999 in the raw PathSnap format. Owner: Member 1.

python -m tests.fixtures.make_mini_trip
6 nodes on a 3 x 2 grid (100 m spacing) near lon 8.65, lat 49.87; 7 two-way streets = 14 arcs
(arc 2k = (a,b), arc 2k+1 = (b,a)); route = arcs 0 -> 10 -> 6 (east, north, east);
14 GPS points with +-3 m noise along the route, a 40 s gap between points 9 and 10,
and one off-road point ~80 m north of the last street (no candidate within 50 m, label -1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

LON0, LAT0 = 8.65, 49.87
M_PER_DEG_LAT = 111_132.0
M_PER_DEG_LON = 111_320.0 * np.cos(np.radians(LAT0))

NODES_XY = [(0, 0), (100, 0), (200, 0), (0, 100), (100, 100), (200, 100)]
STREETS = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
ROUTE = [0, 10, 6]
TRACK_XY = [(10, 3), (30, -2), (50, 2), (70, -3), (90, 1),        # arc 0  (0->1)
            (102, 15), (98, 35), (103, 55), (97, 75), (101, 95),  # arc 10 (1->4)
            (120, 102), (140, 98), (160, 101),                    # arc 6  (4->5)
            (190, 180)]                                           # off-road
TIMES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 51, 52, 53]          # 40 s gap after t=10


def ll(x, y):
    return LON0 + x / M_PER_DEG_LON, LAT0 + y / M_PER_DEG_LAT


def write(dest: Path) -> Path:
    d = dest / "00000999"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "00000999.nodes", "w") as f:
        for x, y in NODES_XY:
            f.write("%.8f %.8f\n" % ll(x, y))
    with open(d / "00000999.arcs", "w") as f:
        for a, b in STREETS:
            f.write(f"{a} {b}\n{b} {a}\n")
    with open(d / "00000999.track", "w") as f:
        for (x, y), t in zip(TRACK_XY, TIMES):
            lon, lat = ll(x, y)
            f.write(f"{lon:.8f} {lat:.8f} {t:.6f}\n")
    with open(d / "00000999.route", "w") as f:
        f.write("\n".join(map(str, ROUTE)) + "\n")
    return d


if __name__ == "__main__":
    print("wrote", write(Path(__file__).resolve().parent / "mini_trip"))
