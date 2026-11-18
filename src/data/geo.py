"""Per-trip projection to metres and arc geometry. Owner: Member 1.

Trips are spread across the world, so each trip uses the UTM zone of its own track.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
from pyproj import Transformer

from src.common.schema import Trip


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


@lru_cache(maxsize=64)
def _fwd(epsg: int) -> Transformer:
    return Transformer.from_crs(4326, epsg, always_xy=True)


@lru_cache(maxsize=64)
def _inv(epsg: int) -> Transformer:
    return Transformer.from_crs(epsg, 4326, always_xy=True)


def to_xy(lon, lat, epsg: int):
    return _fwd(epsg).transform(np.asarray(lon, float), np.asarray(lat, float))


def to_latlon(x, y, epsg: int):
    lon, lat = _inv(epsg).transform(np.asarray(x, float), np.asarray(y, float))
    return lat, lon


def project(trip: Trip, epsg: int | None = None) -> Trip:
    """Return a new Trip with x/y (metres), arc length/bearing and reverse_arc_id filled."""
    if epsg is None:
        epsg = utm_epsg(float(trip.track["lon"].mean()), float(trip.track["lat"].mean()))
    track = trip.track.copy()
    nodes = trip.nodes.copy()
    arcs = trip.arcs.copy()
    track["x"], track["y"] = to_xy(track["lon"], track["lat"], epsg)
    nodes["x"], nodes["y"] = to_xy(nodes["lon"], nodes["lat"], epsg)

    nx, ny = nodes["x"].to_numpy(), nodes["y"].to_numpy()
    u, v = arcs["u"].to_numpy(), arcs["v"].to_numpy()
    dx, dy = nx[v] - nx[u], ny[v] - ny[u]
    arcs["length_m"] = np.hypot(dx, dy)
    arcs["bearing_deg"] = np.degrees(np.arctan2(dx, dy)) % 360.0  # 0 = north, clockwise

    # reverse twin: arc (v,u) for arc (u,v)
    n = len(nodes)
    key = u.astype(np.int64) * n + v
    rkey = v.astype(np.int64) * n + u
    order = np.argsort(key, kind="stable")
    pos = np.searchsorted(key[order], rkey)
    pos = np.clip(pos, 0, len(key) - 1)
    found = key[order][pos] == rkey
    rev = np.where(found, order[pos], -1).astype(np.int64)
    rev[u == v] = -1  # self-loops have no meaningful reverse
    arcs["reverse_arc_id"] = rev
    return replace(trip, track=track, nodes=nodes, arcs=arcs, epsg=int(epsg))


def arc_endpoints(trip: Trip) -> tuple[np.ndarray, np.ndarray]:
    """(A, B) arrays of shape (n_arcs, 2) with start / end coordinates in metres."""
    nx, ny = trip.nodes["x"].to_numpy(), trip.nodes["y"].to_numpy()
    u, v = trip.arcs["u"].to_numpy(), trip.arcs["v"].to_numpy()
    return np.c_[nx[u], ny[u]], np.c_[nx[v], ny[v]]


def point_segment(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise distance from points P to segments A-B and projection parameter t in [0,1]."""
    AB = B - A
    L2 = np.einsum("ij,ij->i", AB, AB)
    t = np.einsum("ij,ij->i", P - A, AB) / np.maximum(L2, 1e-12)
    t = np.clip(t, 0.0, 1.0)
    C = A + t[:, None] * AB
    return np.hypot(P[:, 0] - C[:, 0], P[:, 1] - C[:, 1]), t


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))
