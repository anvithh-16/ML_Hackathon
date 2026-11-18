# Viva notes - Member 1

Everyone must be able to explain the WHOLE pipeline; these are the topics you lead on: projection, alignment DP, loops, candidate radius, every feature, logistic regression, EDA findings.

## My 60-second summary of my part
(write in your own words)

## Likely questions and my answers
1. **Q: Why project per trip instead of using lat/lon?** A: Trips are worldwide; degrees are not metres and a degree
   of longitude shrinks with latitude. Each trip gets its own UTM zone (326xx north / 327xx south), so distances and
   headings are in metres with negligible distortion over a city-sized area.
2. **Q: Why can't label i be route entry i?** A: The route is a chain of arcs (median 5.25 GPS points per arc), not one
   entry per fix. Labels come from a monotonic DP that matches each GPS point to a route *position*.
3. **Q: How do you handle a route that visits the same arc twice?** A: The DP works over positions, not arc ids;
   the two visits are different positions and monotonic order keeps them apart.
4. **Q: Why radius 50 m?** A: Radius sweep on train: 30 m misses 0.31 % of true arcs, 50 m 0.012 % with 9.2 candidates
   per point; 75/100 m add candidates but no recall.
5. **Q: How do you separate a two-way road from its reverse twin?** A: heading_diff_deg: positives median 13°,
   negatives 94°. The LogReg coefficient for it is −1.68 (third largest).
6. **Q: Why GroupKFold by trip?** A: Consecutive points of a trip are almost identical; mixing a trip across train
   and validation would leak and inflate scores.
7. **Q: Why is macro F1 different from Top-1?** A: Top-1 asks whether the best-ranked candidate is right; F1 judges
   every row at threshold 0.5 with ≈ 9 negatives per positive. The balanced variant keeps Top-1 but lowers F1.
8. **Q: Which trips did you exclude and why?** A: 00000054 and 00000033, because their reference routes do not match
   the GPS (7.5 % and 41 % of points could be labelled). Decided on train only.

## Numbers I must know (copy from results/ files, with file names)
- train 80 trips / 208 417 obs; student_test 10 / 19 160 (`results/eda/trip_inventory.csv`)
- 96.7 % of intervals are exactly 1 s; largest gap 153 064 s in 00000090 (`trip_inventory.csv`, `figures/sampling_interval.png`)
- median GPS-to-route distance 4.7 m, p95 ≈ 14 m (`trip_inventory.csv`)
- labelled obs: train 184 009, student_test 18 631; candidate rows 1 842 577 / 201 490 (build_features summary)
- candidate miss rate at 50 m: 0.012 % (`results/candidates/radius_sweep.csv`)
- folds: 5 × 16 trips (`configs/folds.json`); training uses 78 trips, 558 237 rows (stride 3)
- LogReg default: CV Top-1 0.843 ± 0.039, macro F1 0.891; student_test Top-1 0.843, macro F1 0.897, log loss 0.100
  (`results/experiments/logreg__default__20261118-130204.json`)
- nearest-candidate baseline Top-1: 0.540 on student_test
