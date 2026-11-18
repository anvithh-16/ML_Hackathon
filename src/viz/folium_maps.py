"""Folium maps for EDA, label checks and report figures. Owner: Member 1.

trip_map(trip, route_arc_ids=..., pred_arc_ids=...) -> folium.Map with toggleable layers:
Road network (grey) · GPS observations (blue) · Reference route (green) · Predicted route (red)
(+ optional extra predicted layers, e.g. classifier-only vs HMM).
"""
from __future__ import annotations

import folium
import numpy as np
from branca.element import MacroElement, Template
from scipy.spatial import cKDTree

from src.common.schema import Trip
from src.data.geo import project

COLORS = {"network": "#9e9e9e", "gps": "#1565c0", "reference": "#2e7d32", "predicted": "#c62828",
          "extra": ["#ef6c00", "#6a1b9a", "#00838f"]}


def _arc_lines(trip: Trip, arc_ids) -> list[list[list[float]]]:
    """Polylines [[lat,lon],...] for a sequence of arcs (consecutive arcs merged)."""
    n = trip.nodes
    lat, lon = n["lat"].to_numpy(), n["lon"].to_numpy()
    u, v = trip.arcs["u"].to_numpy(), trip.arcs["v"].to_numpy()
    lines, cur = [], []
    for a in arc_ids:
        a = int(a)
        if a < 0:
            if len(cur) > 1:
                lines.append(cur)
            cur = []
            continue
        p0, p1 = [lat[u[a]], lon[u[a]]], [lat[v[a]], lon[v[a]]]
        if cur and cur[-1] == p0:
            cur.append(p1)
        else:
            if len(cur) > 1:
                lines.append(cur)
            cur = [p0, p1]
    if len(cur) > 1:
        lines.append(cur)
    return lines


def _legend(items: list[tuple[str, str]]) -> MacroElement:
    rows = "".join(f'<div><span style="background:{c};width:14px;height:4px;display:inline-block;'
                   f'margin-right:6px;vertical-align:middle"></span>{t}</div>' for t, c in items)
    tpl = Template(f"""{{% macro html(this, kwargs) %}}
<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;
border:1px solid #999;border-radius:4px;font:12px Arial">{rows}</div>{{% endmacro %}}""")
    m = MacroElement()
    m._template = tpl
    return m


def trip_map(trip: Trip, route_arc_ids=None, pred_arc_ids=None, extra_pred: dict | None = None,
             labels=None, show_network: bool = True, network_radius_m: float = 300,
             max_points: int = 2000, title: str | None = None) -> folium.Map:
    if not trip.projected:
        trip = project(trip)
    tr = trip.track
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=15, tiles="OpenStreetMap")
    legend = []

    if show_network:
        nx, ny = trip.nodes.x.to_numpy(), trip.nodes.y.to_numpy()
        tree = cKDTree(tr[["x", "y"]].to_numpy())
        u, v = trip.arcs.u.to_numpy(), trip.arcs.v.to_numpy()
        mid = np.c_[(nx[u] + nx[v]) / 2, (ny[u] + ny[v]) / 2]
        d, _ = tree.query(mid, k=1)
        near = np.where(d <= network_radius_m)[0]
        fg = folium.FeatureGroup(name=f"Road network (≤{int(network_radius_m)} m of track)")
        lat, lon = trip.nodes.lat.to_numpy(), trip.nodes.lon.to_numpy()
        for a in near:
            folium.PolyLine([[lat[u[a]], lon[u[a]]], [lat[v[a]], lon[v[a]]]], color=COLORS["network"],
                            weight=1, opacity=0.6).add_to(fg)
        fg.add_to(m)
        legend.append(("Road network", COLORS["network"]))

    fg = folium.FeatureGroup(name="GPS observations")
    step = max(1, len(tr) // max_points)
    lab = None if labels is None else labels.set_index("obs_idx")["label_arc_id"]
    for r in tr.iloc[::step].itertuples():
        tip = f"obs {r.obs_idx}, t={r.t:g}"
        if lab is not None:
            tip += f", label arc {int(lab.get(r.obs_idx, -1))}"
        folium.CircleMarker([r.lat, r.lon], radius=2, color=COLORS["gps"], fill=True, opacity=0.8,
                            tooltip=tip).add_to(fg)
    fg.add_to(m)
    legend.append(("GPS observations", COLORS["gps"]))

    if route_arc_ids is not None:
        fg = folium.FeatureGroup(name="Reference route")
        for line in _arc_lines(trip, route_arc_ids):
            folium.PolyLine(line, color=COLORS["reference"], weight=5, opacity=0.7).add_to(fg)
        fg.add_to(m)
        legend.append(("Reference route", COLORS["reference"]))
    if pred_arc_ids is not None:
        fg = folium.FeatureGroup(name="Predicted route")
        for line in _arc_lines(trip, pred_arc_ids):
            folium.PolyLine(line, color=COLORS["predicted"], weight=3, opacity=0.9).add_to(fg)
        fg.add_to(m)
        legend.append(("Predicted route", COLORS["predicted"]))
    for i, (name, arcs) in enumerate((extra_pred or {}).items()):
        col = COLORS["extra"][i % len(COLORS["extra"])]
        fg = folium.FeatureGroup(name=name, show=False)
        for line in _arc_lines(trip, arcs):
            folium.PolyLine(line, color=col, weight=3, opacity=0.9, dash_array="6").add_to(fg)
        fg.add_to(m)
        legend.append((name, col))

    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    if title:
        legend.insert(0, (f"<b>{title}</b>", "transparent"))
    m.get_root().add_child(_legend(legend))
    return m
