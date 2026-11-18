# 2.5 Labels, candidates and features  (owner: Member 1)

Reproduce: `python -m src.features.build_features --partition train` (and `--partition student_test`),
`python -m src.labels.check_alignment`, `python -m src.candidates.generate --sweep`.

## Label construction (route-to-GPS alignment)

The `.route` file is an ordered list of arc ids (median 5.25 GPS points per route arc), so row *i* of the track
is not row *i* of the route. Each GPS point is therefore matched to a **position** in the route with a monotonic
dynamic programme (`src/labels/align_route.py`):

1. Route arcs are densified and indexed in a KD-tree; for every GPS point the route positions within 150 m are its
   options.
2. Moving from position *j* to *j′* is allowed when 0 ≤ *j′* − *j* ≤ 15, at a cost of 2 m per skipped position, on
   top of the point-to-arc distance. If no option is reachable (e.g. after a long gap) any forward jump is allowed;
   if even that fails, the alignment restarts with a large penalty.
3. Backtracking gives one route position per point; its arc is the label. Points with no route position within
   150 m, or aligned farther than 50 m, get label −1 and are excluded from training and scoring.

Working over **positions** rather than arc ids is what handles loops: if a route passes the same arc twice, the
two visits are different positions, and the monotonic order assigns each GPS point to the right visit.

**Evidence it works** (`results/labels/alignment_summary.csv`, 90 trips): the median trip labels 100 % of its points
with a median aligned distance of 4.7 m (95th percentile 14.5 m), in line with the GPS noise in §2.3. Labelled
observations: train 184 009 of 208 417, student_test 18 631 of 19 160.

Trips flagged by the checker (< 95 % labelled or a non-monotonic restart), reviewed on the overlay maps in
`results/labels/maps/align_<trip>.html`:

| trip | labelled | median / p95 dist | backward restarts | decision |
|---|---|---|---|---|
| train 00000054 | 7.5 % | 136 / 149 m | yes | excluded from training |
| train 00000033 | 41 % | 122 / 142 m | no | excluded from training |
| train 00000002 | 88 % | 5.3 / 61 m | no | kept: two off-route stretches (552 and 311 points) get −1 |
| train 00000043 | 93 % | 10.8 / 58 m | no | kept: two short off-route stretches |
| train 00000066 | 90 % | 6.8 / 60 m | 5 | kept: a few small restarts |
| train 00000081 | 79 % | 6.0 / 143 m | 138 | **to decide** — many restarts; only 517 of 778 route positions covered |
| student_test 00000056 | 87 % | 5.7 / 134 m | 132 | kept (test trips are not removed); noted as a hard trip |

## Candidate generation

For every GPS point (`src/candidates/generate.py`) all arcs within **50 m** are candidates (arcs densified every 10 m
into a KD-tree, then exact point-to-segment distance). If none is within 50 m, the 3 nearest arcs within 200 m are
used instead. At most 30 candidates per point, ranked by distance.

The radius was chosen on **train** only (`results/candidates/radius_sweep.csv`); miss rate = labelled points whose
true arc is not among the candidates:

| radius | miss rate | candidates / obs |
|---|---|---|
| 30 m | 0.309 % | 5.9 |
| **50 m** | **0.012 %** | **9.2** |
| 75 m | 0.012 % | 13.2 |
| 100 m | 0.012 % | 16.2 |

50 m removes almost all misses; larger radii only add negatives. Final tables: train 1 842 577 candidate rows
(miss rate 0.023 % over the full train set), student_test 201 490 rows (0.005 %). About 9–10 negatives per positive.

## Features (FEATURE_VERSION v1) and why each was chosen

All features use only the GPS track and the network, never the route.

| feature | definition | why it helps |
|---|---|---|
| dist_m | distance from the GPS point to the arc (m) | the true arc is usually close (median 4.9 m for positives vs 25 m overall, train) |
| dist_rank | rank of the arc by distance within the point's candidates (0 = nearest) | relative closeness; the nearest arc is right only 56 % of the time (train) |
| dist_minus_min | dist_m minus the nearest candidate's distance | how much farther this arc is than the best alternative |
| heading_diff_deg | angle (0–180°) between the vehicle heading (from neighbouring fixes) and the arc bearing; 90 when heading is invalid | separates a road from its reverse twin: positives median 13°, negatives 94° (train) |
| heading_valid | 1 if the neighbouring fixes are ≥ 2 m apart | heading is noise when the vehicle stands still |
| speed_mps | displacement / time between neighbouring fixes | speed hints at road type (motorway vs side street) |
| dt_prev_s | time since the previous fix | marks points after gaps, where neighbour features are unreliable |
| arc_length_m | length of the arc | very short arcs (junction connectors) behave differently |
| proj_t | position of the point's projection along the arc (0 = start, 1 = end) | points projecting onto an arc's end are often at junctions |
| n_candidates | number of candidates of this point | dense areas are harder; calibrates the other features |
| u_degree, v_degree | number of arcs touching the arc's start / end node | junction complexity |
| prev_obs_dist_same_arc, next_obs_dist_same_arc | distance from the previous / next GPS point to the same arc (capped at 100 m) | consistency over time: a correct arc stays close to the neighbours too |
| reverse_exists | 1 if the arc has a reverse twin | tells the model when heading is decisive |

All columns are float32 and checked for NaN/inf (`tests/test_features.py`).

## Leakage controls

- Features never read the route: `tests/test_features.py` builds features with and without labels and checks they are
  identical; `load_route` is called only to build labels.
- Cross-validation is grouped by **trip** (`configs/folds.json`, 5 folds of 16 trips), so no trip is in both the
  training and the validation folds.
- student_test is used only for comparing models, never for training, tuning or threshold choice.
