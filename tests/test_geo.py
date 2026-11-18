"""Projection tests. Owner: Member 1."""
import numpy as np

from src.data.geo import to_xy, utm_epsg


def test_distance_scale():
    e = utm_epsg(8.65, 49.87)
    x, y = to_xy([8.65, 8.65], [49.870, 49.871], e)
    assert abs(np.hypot(x[1] - x[0], y[1] - y[0]) - 111.2) < 1.5


def test_hemispheres():
    assert 32601 <= utm_epsg(8.6, 49.8) <= 32660
    assert 32701 <= utm_epsg(-58.45, -34.62) <= 32760   # Buenos Aires (trip 00000080)


def test_reverse_twins(mini_trip):
    a = mini_trip.arcs
    for r in a.itertuples():
        assert r.reverse_arc_id >= 0
        rv = a.loc[r.reverse_arc_id]
        assert (rv.u, rv.v) == (r.v, r.u)
    assert np.allclose(a.length_m, 100, atol=1.5)
