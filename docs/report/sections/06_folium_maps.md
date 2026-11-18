# 6. Geographical visualisation (Folium)  (owner: Member 1)

All maps are made by one function, `src/viz/folium_maps.trip_map`, so every map in the report looks the same.

**Layers** (each can be toggled in the layer control):
- **Road network** — arcs within 300 m of the GPS track, thin grey lines (a full network can have 300 000 arcs,
  which would freeze the browser).
- **GPS observations** — small blue circle markers, thinned to at most 2 000 points per map; the tooltip shows the
  observation index, time and (on alignment maps) its label arc, so −1 points can be found by hovering.
- **Reference route** — thick green line (only on development maps; never sent to the UI).
- **Predicted route** — red line from the HMM decoder.
- **Extra predictions** (optional) — dashed orange / purple / teal lines, hidden by default, e.g. classifier-only vs
  classifier + HMM.

**Checklist:** distinguishable colours (grey network, blue GPS, green reference, red prediction, dashed extras), a fixed legend,
`LayerControl`, and `fit_bounds` on the track so the map opens at a sensible zoom.

**Committed interactive maps**
- `results/eda/maps/` — 3 train trips (00000001, 00000041, 00000090) + 1 student_test trip (00000000): GPS, network,
  reference.
- `results/labels/maps/` — alignment overlays for 00000001, 00000010, 00000041, 00000075 (normal) and every flagged
  trip (00000002, 00000033, 00000043, 00000054, 00000066, 00000081, student_test 00000056).
- `results/hmm/maps/` — predicted vs reference route (from M3's `compare_hmm`, once merged).

**Representative static overlays (shown below and in §2.5):**
1. `align_00000001.html` — a normal trip: labelled points sit on the reference route.
2. `align_00000054.html` — the excluded trip: GPS and reference route visibly diverge (only 7.5 % of points labelled).
3. A predicted-vs-reference route from `results/hmm/maps/`.
