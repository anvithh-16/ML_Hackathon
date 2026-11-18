# 2. Dataset understanding and EDA  (owner: Member 1)

Every number below is read from a file in `results/` (named in brackets). Partition is stated for every analysis.
Reproduce: `python -m src.data.validate`, `python -m eda.eda_plots`, `python -m eda.eda_maps`.

## 2.1 How the raw data is organised

The organiser folder `Student folder/` has two partitions, `train/` (80 trips) and `student_test/` (10 trips).
Each trip is a folder `<trip_id>/` with four whitespace-separated, header-less text files:

| file | columns | meaning |
|---|---|---|
| `.track` | lon, lat, t | one GPS fix per row, t in seconds |
| `.nodes` | lon, lat | road-network vertices; row index = node id (0-based) |
| `.arcs` | from_node, to_node | directed road segments; row index = arc id (0-based) |
| `.route` | arc_id | reference route as an ordered chain of arc ids (ground truth) |

Arcs are **directed**: a two-way street is stored as two arcs (u→v and v→u). The median trip has 95.1 % of its arcs
with such a reverse twin (`reverse_arc_frac`, range 35–100 %), so heading is the only thing that separates the
correct arc from its twin.

Inventory (`results/eda/trip_inventory.csv`):

| partition | trips | GPS observations | nodes | arcs | route entries |
|---|---|---|---|---|---|
| train | 80 | 208 417 | 1 933 485 | 3 876 904 | 31 336 |
| student_test | 10 | 19 160 | 90 689 | 185 041 | 3 477 |

Per trip (train + student_test): 262–19 313 observations (median 1 497), 299–155 003 nodes (median 9 305),
649–307 392 arcs (median 18 762), 59–2 266 route entries (median 273).

## 2.2 Preprocessing

- **Per-trip UTM projection.** Trips lie between longitude −100° and 137° and latitude −35° and 62°, so there is no
  single metric projection. Each trip is projected to its own UTM zone, chosen from the mean track position
  (EPSG 326xx north, 327xx south; 3 trips are southern-hemisphere). After projection every distance and heading is
  in metres/degrees on a flat plane. Arc length, bearing and reverse twin are computed once per trip (`src/data/geo.py`).
- **Integrity checks** (`src/data/validate.py`, all 90 trips): all four files present; columns numeric; lon/lat in
  range; arc endpoints and route ids within the node/arc tables; timestamps strictly increasing.
- **Missing values:** 0 in every trip (`missing_values` column). No timestamp needed repairing (`time_repaired` = 0),
  although the loader would sort/deduplicate them for evaluator trips.
- Other findings: 24 self-loop arcs across 10 trips (kept, no reverse twin), up to 11 duplicate node coordinates.

## 2.3 Patterns observed

- **Sampling and gaps** (`figures/sampling_interval.png`, train + student_test): 96.7 % of consecutive fixes are exactly
  1 s apart, but 30 trips contain at least one gap > 30 s. Extreme cases: 00000090 (153 064 s ≈ 42 h),
  00000078 (52 810 s), 00000069 (2 658 s), 00000021 (2 161 s). These gaps split the HMM into independent segments
  (`gap_split_s = 30`) and `dt_prev_s` is capped in the features.
- **GPS rows ≠ route entries** (`figures/obs_vs_route_len.png`): the route is a list of arcs, not one entry per GPS
  fix. A route arc receives 1.3–57 GPS points (median 5.25, `obs_per_route_arc`), so labels have to come from an
  alignment (§2.5), not from row positions.
- **GPS noise** (`figures/gps_to_route_distance.png`, train, excluded trips removed): median distance from a GPS
  point to the reference route is 4.7 m per trip; the per-trip 95th percentile has a median of 14.2 m.
- **Network size and connectivity** (`figures/network_size.png`, `network_connectivity.csv`): arcs grow linearly
  with nodes (≈ 2 arcs per node, consistent with two-way roads). Networks are clipped around the trip, so they
  split into 1–421 weakly connected pieces (median 19), but the largest component holds a median 98 % of the nodes.
- **Candidates per observation** (`figures/candidates_per_obs.png`, train, radius 50 m): mostly 3–10, capped at 30.
  The histogram alternates between even and odd counts because reverse twins arrive in pairs.

### Anomalies and decisions (`results/eda/anomalies.csv`, 67 rows)

| rule | trips | decision |
|---|---|---|
| gap_gt_split | 30 | kept; HMM splits into segments; dt capped |
| low_1hz_fraction (< 90 % at 1 s) | 14 | kept |
| route_discontinuity | 11 | kept; labels come from alignment, not route order |
| self_loops | 9 | kept |
| gps_far_from_route (> 10 % of GPS > 50 m from route) | 00000054, 00000033, 00000002 | see below |

- **00000054** (87 % of GPS > 50 m from its route) and **00000033** (only 41 % of points could be labelled; median
  aligned distance 122 m) — the reference route does not describe the drive. **Excluded from training**
  (`exclude_trips_train` in `configs/base.yaml`); training uses 78 trips.
- **00000002** (12 % of GPS > 50 m): the far points form two contiguous stretches at which the GPS leaves the route;
  the rest aligns well (median 4.9 m). **Kept**; the far points simply get label −1.

## 2.4 Maps

Folium maps with GPS track, road network (within 300 m of the track) and reference route, with legend and layer
control: `results/eda/maps/train_00000001.html`, `train_00000041.html`, `train_00000090.html` (the 42-hour gap trip)
and `student_test_00000000.html`. The PDF shows static overlays of the same data (GPS, reference route, label status).
