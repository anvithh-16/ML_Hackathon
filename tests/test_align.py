"""Alignment tests. Owner: Member 1."""
import numpy as np

from src.data.loader import load_route
from src.labels.align_route import align


def test_mini_alignment(mini_trip, mini_dir, cfg):
    lab = align(mini_trip, load_route(mini_dir), cfg)
    assert list(lab.label_arc_id[:5]) == [0] * 5
    assert list(lab.label_arc_id[5:10]) == [10] * 5
    assert list(lab.label_arc_id[10:13]) == [6] * 3
    assert lab.label_arc_id.iloc[13] == -1                 # off-road point
    rp = lab.route_pos[lab.route_pos >= 0].to_numpy()
    assert (np.diff(rp) >= 0).all()


def test_loop_route_labels_both_visits(mini_trip, cfg):
    # route goes 0 (0->1), 10 (1->4), 5 (4->3), 9 (3->0), 0 again: arc 0 visited twice
    route = np.array([0, 10, 5, 9, 0])
    from dataclasses import replace
    import pandas as pd
    xy = [(20, 2), (60, -1), (100, 30), (100, 70), (70, 100), (30, 100), (0, 70), (0, 30), (20, -2), (60, 1)]
    tr = mini_trip.track.iloc[:0].copy()
    tr = pd.DataFrame({"obs_idx": range(len(xy)), "lon": 0.0, "lat": 0.0, "t": np.arange(len(xy), dtype=float),
                       "x": [p[0] + mini_trip.nodes.x[0] for p in xy], "y": [p[1] + mini_trip.nodes.y[0] for p in xy]})
    trip = replace(mini_trip, track=tr)
    lab = align(trip, route, cfg)
    assert list(lab.route_pos[:2]) == [0, 0]
    assert list(lab.route_pos[-2:]) == [4, 4]
