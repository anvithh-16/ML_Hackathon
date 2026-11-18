"""Road graph tests on the mini grid. Owner: Member 3."""
import numpy as np

from src.hmm.graph import RoadGraph


def test_network_distance(mini_trip):
    g = RoadGraph(mini_trip)
    # from middle of arc 0 (0->1) to middle of arc 10 (1->4): 50 + 50
    assert abs(g.network_distance(0, 0.5, 10, 0.5, 1000) - 100) < 2
    # arc 0 middle to arc 6 (4->5) middle: 50 + 100 + 50
    assert abs(g.network_distance(0, 0.5, 6, 0.5, 1000) - 200) < 3
    assert np.isinf(g.network_distance(0, 0.5, 6, 0.5, 120))       # beyond the limit
    assert g.shortest_arc_path(0, 6, 1000) == [10]
    assert g.connected(0, 10) and not g.connected(0, 6)
