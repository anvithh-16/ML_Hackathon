"""Route metric tests. Owner: Member 3."""
import numpy as np

from src.eval.route_metrics import route_continuity, route_precision_recall

u = np.array([0, 1, 1, 4])   # arcs: 0:0->1, 1:1->4, 2:1->9, 3:4->5
v = np.array([1, 4, 9, 5])


def test_continuity():
    assert route_continuity([0, 0, 1, 3], u, v) == 1.0
    assert abs(route_continuity([0, 3, 3], u, v) - 0.5) < 1e-12
    assert np.isnan(route_continuity([0], u, v))


def test_precision_recall():
    r = route_precision_recall([[0, 1, 2]], np.array([0, 1, 3]))
    assert abs(r["route_precision"] - 2 / 3) < 1e-12 and abs(r["route_recall"] - 2 / 3) < 1e-12
