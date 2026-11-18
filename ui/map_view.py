"""Folium map built from an API response. Owner: Member 2. No imports from src/."""
from __future__ import annotations

import folium
from branca.element import MacroElement, Template

GPS, PRED = "#1565c0", "#c62828"


def prediction_map(resp: dict, max_markers: int = 1500) -> folium.Map:
    gps = resp.get("gps_latlon") or []
    lines = resp.get("predicted_route_latlon") or []
    pts = gps + [p for line in lines for p in line]
    if not pts:
        return folium.Map(location=[0, 0], zoom_start=2)
    lats, lons = [p[0] for p in pts], [p[1] for p in pts]
    m = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], zoom_start=15, tiles="OpenStreetMap")
    fg = folium.FeatureGroup(name="GPS observations")
    step = max(1, len(gps) // max_markers)
    for lat, lon in gps[::step]:
        folium.CircleMarker([lat, lon], radius=2, color=GPS, fill=True, opacity=0.8).add_to(fg)
    fg.add_to(m)
    fg = folium.FeatureGroup(name="Predicted route")
    for i, line in enumerate(lines):
        folium.PolyLine(line, color=PRED, weight=4, opacity=0.9, tooltip=f"segment {i}").add_to(fg)
    fg.add_to(m)
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    folium.LayerControl(collapsed=False).add_to(m)
    legend = MacroElement()
    legend._template = Template(f"""{{% macro html(this, kwargs) %}}
<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;
border:1px solid #999;border-radius:4px;font:12px Arial">
<div><span style="background:{GPS};width:14px;height:4px;display:inline-block;margin-right:6px"></span>GPS observations</div>
<div><span style="background:{PRED};width:14px;height:4px;display:inline-block;margin-right:6px"></span>Predicted route</div>
</div>{{% endmacro %}}""")
    m.get_root().add_child(legend)
    return m
