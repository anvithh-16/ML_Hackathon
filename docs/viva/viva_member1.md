# Viva notes - Member 1

Everyone must be able to explain the WHOLE pipeline; these are the topics I lead on: projection, alignment DP, loops,
candidate radius, every feature, logistic regression, EDA findings.

## My 60-second summary of my part
*(Draft: re-read it, change it to how you would say it, and make sure you can defend every sentence.)*

I built the data foundation that the other three members and the API depend on. Each trip is loaded and projected to
its own UTM zone, because the trips are spread across the world and degrees are not metres. The reference route is a
list of arcs, not one entry per GPS point, so I build the labels with a monotonic dynamic programme that aligns every
GPS point to a position along the route; working over positions instead of arc ids also handles routes that loop. For
every GPS point I generate candidate arcs within 50 m (0.012 % of true arcs missed), and compute 15 features per
(point, arc) pair — distance, heading difference, neighbour consistency and so on — that never read the route.
That gives 208 417 train observations (1.84 M candidate rows) and 19 160 student_test observations. Folds are grouped
by trip so that no trip is in both training and validation. I also trained Logistic Regression as the linear baseline:
Top-1 0.843 against 0.540 for the nearest-road baseline, with cross-validation agreeing with the test result. I did the EDA, the Folium maps and the README, and I assembled the report.

## Likely questions and my answers
1. **Q: Why project per trip instead of using lat/lon?** A: Trips are worldwide; a degree of longitude shrinks with
   latitude and degrees are not metres. Each trip gets its own UTM zone (326xx north / 327xx south), so distances and
   headings are in metres with negligible distortion over a city-sized area.
2. **Q: Why can't label i be route entry i?** A: The route is a chain of arcs with a median of 5.25 GPS points per arc,
   not one entry per fix. Labels come from a monotonic DP that matches each GPS point to a route *position*.
3. **Q: Explain the alignment DP.** A: States are route positions (densified route arcs). A point may stay or move
   forward by up to 15 positions, paying 2 m per skipped position on top of its distance to that position. After a time
   gap any forward jump is allowed; if nothing is reachable it restarts with a large penalty. Backtracking gives one
   position per point; points farther than 50 m from their aligned position (or with none within 150 m) get label −1.
4. **Q: How do you handle a route that visits the same arc twice?** A: The DP works over positions, not arc ids;
   the two visits are different positions and monotonic order keeps them apart.
5. **Q: Why radius 50 m?** A: Radius sweep on train only: 30 m misses 0.31 % of true arcs, 50 m misses 0.012 % with
   9.2 candidates per point; 75 m and 100 m add candidates but no recall.
6. **Q: How do you separate a two-way road from its reverse twin?** A: heading_diff_deg: positives have a median of 13°,
   negatives 94°. Its Logistic Regression coefficient is −1.68 (third largest after dist_rank and dist_minus_min).
7. **Q: Why GroupKFold by trip?** A: Consecutive points of a trip are almost identical; mixing a trip across training
   and validation would leak and inflate scores.
8. **Q: Why is Macro F1 different from Top-1?** A: Top-1 asks whether the best-ranked candidate of each point is right;
   F1 judges every row at threshold 0.5 with about 9 negatives per positive. The balanced Logistic Regression keeps
   Top-1 (0.844) but its Macro F1 drops from 0.897 to 0.815.
9. **Q: Which trips did you exclude and why?** A: 00000054 and 00000033, because their reference routes do not match
   the GPS (7.5 % and 41 % of points could be labelled; median aligned distance 136 m and 122 m). Decided on train only.
10. **Q: What about 00000081 and student_test 00000056, which the checker flagged?** A: I looked at the overlays.
    In 00000081 the GPS leaves the reference route for one long stretch (those points get −1) and the vehicle finishes
    by driving back past the start of the loop, so the last ~400 points align to early route positions (15 backward
    steps among labelled points, not label noise). I kept it. 00000056 is a test trip, and test trips are never removed.
11. **Q: How do you know the labels are right?** A: median aligned distance is 4.7 m (p95 14.5 m), matching the GPS
    noise; unit tests on the mini trip incl. a loop route; overlay maps reviewed; and a nearest-candidate baseline of
    0.54 Top-1 (neither near 0 nor near 1).
12. **Q: Why does Logistic Regression underfit here?** A: One linear boundary cannot express interactions (heading
    matters mainly when the vehicle is moving); the tree models gain about 2 points of Top-1 over it.
13. **Q: Which model did the team select and why?** A: XGBoost default: highest student_test Top-1 (0.862), lowest log
    loss (0.0692) and fast (0.47 ms/1k rows); Optuna tuning gained only 0.0026 CV Macro F1 and did not improve Top-1.

## Numbers I must know (copy from results/ files, with file names)
- train 80 trips / 208 417 obs; student_test 10 / 19 160 (`results/eda/trip_inventory.csv`)
- 96.7 % of time intervals are exactly 1 s; largest gap 153 064 s in 00000090 (`trip_inventory.csv`, `figures/sampling_interval.png`)
- median GPS-to-route distance 4.7 m, p95 ≈ 14 m (`trip_inventory.csv`)
- labelled obs: train 184 009, student_test 18 631; candidate rows 1 842 577 / 201 490 (build_features summary)
- candidate miss rate at 50 m: 0.012 % (`results/candidates/radius_sweep.csv`)
- folds: 5 × 16 trips (`configs/folds.json`); training uses 78 trips, 558 237 rows with stride 3
- LogReg default: CV Top-1 0.843 ± 0.039, CV Macro F1 0.891; student_test Top-1 0.843, Macro F1 0.897, log loss 0.100
  (`results/experiments/logreg__default__20261118-130204.json`)
- nearest-candidate baseline Top-1: 0.540 on student_test (`results/model_comparison.csv`)
- selected model xgb__default: Top-1 0.862, log loss 0.0692; with HMM: Top-1 0.942, route continuity 0.979 (`results/hmm/clf_vs_hmm.csv`)
