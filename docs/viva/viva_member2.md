# Viva notes - Member 2

Everyone must be able to explain the WHOLE pipeline; these are the topics I lead on: Top-1 vs macro F1 vs log loss, GroupKFold vs random split leakage, KNN cost, decision-tree overfitting, UI flow.

## My 60-second summary of my part

I built the part that makes the model comparison fair: one metrics file, one training and evaluation runner and one run
record per experiment, so all five models use the same folds, the same subsample and the same metrics. I also wrote the KNN
and Decision Tree models, the comparison table and plots, and the Streamlit evaluator interface. We cast map matching as a
yes/no question for each (GPS point, candidate arc) pair and judge it mainly by Top-1: the share of GPS points where the
highest-scoring candidate is the true arc. Cross-validation is grouped by trip so no trip is in both training and validation.
On student_test, KNN gets Top-1 0.832 and the Decision Tree 0.851, against 0.540 for the nearest-candidate baseline.
KNN is about 9,500 times slower at prediction than the tree (368 ms vs 0.039 ms per 1,000 rows). The tree overfits mildly:
about 2 points better on training folds than validation folds. XGBoost was best overall (Top-1 0.862) and the HMM lifts it
further. The UI shows one trip at a time and never receives the reference route.

## Likely questions and my answers

1. **Q: Top-1 vs macro F1 vs log loss?**
   A: Top-1 is per GPS point: among that point's candidates, is the highest-scoring one the true arc? It matches what we need
   (one arc per point). Macro F1 treats each candidate row as correct/wrong and averages the F1 of both classes; it is higher
   than Top-1 because the many easy "wrong arc" rows are counted. Log loss judges the probabilities, not just the ranking:
   confident wrong answers are punished heavily. A model can have a good Top-1 and a poor log loss (KNN).

2. **Q: Why GroupKFold by trip and not a random split?**
   A: Consecutive GPS points of one trip are nearly identical, and their candidates share the same arcs. In a random split,
   neighbours of a validation row sit in the training set, so the score is inflated (leakage). Grouping by trip means
   validation always uses whole unseen trips, like the evaluator's trips. `configs/folds.json` fixes the folds and
   `tests/test_no_leakage.py` checks them; the runner asserts train and validation trips are disjoint.

3. **Q: Why is KNN slow and does it matter?**
   A: It stores all 558,237 training rows and compares every new row with them: 367.6 ms per 1,000 rows (about 33 MB
   artifact) against 0.039 ms for the tree. It matters because the API scores every candidate of a trip, and KNN also had the
   lowest Top-1, so it was not worth the cost.

4. **Q: What is your evidence that the tree overfits?**
   A: `results/eval/dtree_train_vs_val.csv`: mean training Top-1 0.867 vs validation 0.846 (gap 0.022). The depth sweep
   (`dtree_depth_sweep.csv`) shows the training score and training log loss keep improving with depth, validation Top-1 stops
   improving at depth 12 and validation log loss gets worse (0.097 at depth 12, 0.119 unlimited). It is mild because
   `min_samples_leaf` is 50. I must also say that two folds had validation above training because those trips are easier.

5. **Q: Why scale features for KNN but not for the tree?**
   A: KNN uses distances, so a feature with large numbers would dominate. A tree only compares one feature with a threshold,
   so scale does not matter.

6. **Q: Why did you keep the untuned XGBoost?**
   A: We declared student_test Top-1 as the main selection metric. Tuning with Optuna raised CV macro F1 slightly (0.9151 vs
   0.9125) but lowered student_test Top-1 (0.858 vs 0.862) and doubled inference time (`results/tuning/tuned_vs_untuned.csv`),
   so we kept the untuned model. (Optuna and the export are Member 4's work; I ran the tuned variant through my runner.)

7. **Q: How does the UI work and where is the ground truth?**
   A: Streamlit calls `/health`, `/trips`, then `/predict/<trip>` on the API laptop. The API runs candidates, the model and the
   HMM, and computes the metrics itself against the `.route` file. The UI gets only GPS points, the predicted route and the
   metrics, never the reference route. Errors become an `ApiError` and a red box.

8. **Q: Why compare with a baseline?**
   A: To show the models add something. Picking the nearest candidate gets Top-1 0.540, the models reach 0.83 to 0.86.

9. **Q: What does the HMM add?**
   A: It turns independent per-point choices into a connected route. With XGBoost scores, Top-1 (mean over trips) goes from
   0.880 to 0.942 and route continuity from 0.935 to 0.979 (`results/hmm/clf_vs_hmm.csv`).

10. **Q: What are the weak points of your work?**
    A: The models use every 3rd training observation. The decision-tree depth conclusion is within fold noise (std about 0.04).
    KNN training-fold accuracy was not measured (the training row is its own neighbour). The route-continuity numbers for
    Logistic Regression and Random Forest in the comparison come from score files I regenerated locally, which differ slightly
    from the recorded runs. The UI was shown on one laptop; the two-laptop test was not done.

## Numbers I must know (copy from results/ files, with file names)

| What | Value | File |
|---|---|---|
| Training rows / trips (stride 3) | 558,237 / 78 | `results/experiments/knn__default__*.json` |
| KNN student_test Top-1 / macro F1 / log loss | 0.8321 / 0.8997 / 0.1128 | same |
| KNN CV Top-1 / macro F1 | 0.8353 ± 0.0378 / 0.8969 ± 0.0208 | same |
| KNN inference time | 367.6 ms per 1,000 rows | same |
| Decision Tree student_test Top-1 / macro F1 / log loss | 0.8513 / 0.9154 / 0.0803 | `results/experiments/dtree__default__*.json` |
| Decision Tree CV Top-1 / macro F1 | 0.8459 ± 0.0427 / 0.9086 ± 0.0227 | same |
| Decision Tree inference time | 0.039 ms per 1,000 rows | same |
| Tree training vs validation Top-1 (mean) | 0.8674 vs 0.8459 | `results/eval/dtree_train_vs_val.csv` |
| Tree validation log loss at depth 12 vs unlimited | 0.0965 vs 0.1192 | `results/eval/dtree_depth_sweep.csv` |
| Nearest-candidate baseline Top-1 | 0.5399 | `results/model_comparison.csv` |
| XGBoost default student_test Top-1 / log loss | 0.8621 / 0.0692 | `results/model_comparison.csv` |
| Random Forest student_test Top-1 | 0.8557 | same |
| XGBoost tuned vs default Top-1 | 0.8580 vs 0.8621 | `results/tuning/tuned_vs_untuned.csv` |
| HMM vs classifier-only Top-1 (XGBoost, trip mean) | 0.9423 vs 0.8798 | `results/hmm/clf_vs_hmm.csv` |
| UI demo trip 00000007 (HMM Top-1 / round trip) | 99.34% / 0.2 s | `results/ui/ui_demo_00000007.json` |
