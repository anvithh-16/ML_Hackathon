"""Road graph with bounded, memoised shortest paths. Owner: Member 3."""
from __future__ import annotations

import heapq

import numpy as np

from src.common.schema import Trip
from src.data.geo import project


class RoadGraph:
    def __init__(self, trip: Trip):
        if not trip.projected:
            trip = project(trip)
        a = trip.arcs
        self.u = a["u"].to_numpy(np.int64)
        self.v = a["v"].to_numpy(np.int64)
        self.length = a["length_m"].to_numpy(float)
        self.rev = a["reverse_arc_id"].to_numpy(np.int64)
        n = len(trip.nodes)
        order = np.argsort(self.u, kind="stable")
        self.out_arcs = order                       # arcs sorted by start node
        self.indptr = np.searchsorted(self.u[order], np.arange(n + 1))
        self._cache: dict[int, tuple[float, dict]] = {}

    def outgoing(self, node: int) -> np.ndarray:
        return self.out_arcs[self.indptr[node]:self.indptr[node + 1]]

    def dijkstra(self, src: int, limit_m: float) -> dict[int, tuple[float, int]]:
        """node -> (distance, incoming arc) for all nodes within limit_m of src. Memoised."""
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= limit_m:
            return hit[1]
        dist = {src: (0.0, -1)}
        heap = [(0.0, src)]
        done = set()
        while heap:
            d, node = heapq.heappop(heap)
            if node in done:
                continue
            done.add(node)
            if d > limit_m:
                break
            for arc in self.outgoing(node):
                nd = d + self.length[arc]
                w = int(self.v[arc])
                if nd <= limit_m and (w not in dist or nd < dist[w][0]):
                    dist[w] = (nd, int(arc))
                    heapq.heappush(heap, (nd, w))
        self._cache[src] = (limit_m, dist)
        return dist

    def network_distance(self, arc_a: int, t_a: float, arc_b: int, t_b: float, limit_m: float) -> float:
        """Driving distance from position t_a on arc_a to t_b on arc_b (inf if beyond limit)."""
        if arc_a == arc_b:
            return abs(t_b - t_a) * self.length[arc_a]   # GPS noise may move 'backwards' slightly
        head = (1.0 - t_a) * self.length[arc_a]
        tail = t_b * self.length[arc_b]
        if self.v[arc_a] == self.u[arc_b]:
            return head + tail
        rem = limit_m - head - tail
        if rem < 0:
            return np.inf
        d = self.dijkstra(int(self.v[arc_a]), max(rem, 0.0))
        hit = d.get(int(self.u[arc_b]))
        return np.inf if hit is None or hit[0] > rem else head + hit[0] + tail

    def shortest_arc_path(self, arc_a: int, arc_b: int, limit_m: float) -> list[int]:
        """Arcs strictly between arc_a and arc_b on the shortest path ([] if adjacent/unreachable)."""
        if arc_a == arc_b or self.v[arc_a] == self.u[arc_b]:
            return []
        d = self.dijkstra(int(self.v[arc_a]), limit_m)
        node = int(self.u[arc_b])
        if node not in d:
            return []
        path = []
        while True:
            _, inc = d[node]
            if inc < 0:
                break
            path.append(inc)
            node = int(self.u[inc])
            if len(path) > 10000:
                break
        return path[::-1]

    def connected(self, a: int, b: int) -> bool:
        return a == b or self.v[a] == self.u[b]


def build_graph(trip: Trip) -> RoadGraph:
    return RoadGraph(trip)
