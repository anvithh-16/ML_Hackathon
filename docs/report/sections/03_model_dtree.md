# 3.2 Decision Tree  (owner: Member 2)

Run record: `results/experiments/dtree__default__20261118-130319.json` · Config: `configs/models/dtree.yaml` · Code: `src/models/dtree.py`
Overfitting evidence: `results/eval/dtree_train_vs_val.csv` and `results/eval/dtree_depth_sweep.csv`, both produced by
`python -m src.eval.train_vs_val --model dtree --depths 3 6 12 20 30 0`.

## A. Model understanding

**1. Why is this algorithm suitable for PathSnap?**
The model answers "is this candidate arc the correct one?" for every (GPS point, candidate arc) pair. A decision tree does
this with a chain of yes/no questions on the features, such as "is the distance below some threshold?" followed by "is the
heading difference small?". That matches how we would reason by hand: the correct arc is usually close to the GPS point and
points in the direction of travel. The tree can combine features in a non-linear way, needs no feature scaling, and its rules
can be read and explained. It is also very fast, which matters because the API has to score every candidate of a trip.

**2. Strengths and limitations for this task.**
- *Strength, speed:* prediction is a handful of comparisons per row, measured at 0.039 ms per 1,000 candidate rows.
- *Strength, interpretable:* we can inspect which features the splits use.
- *Limitation, coarse probabilities:* the score is the share of correct arcs among the training rows in one leaf, so all rows
  in a leaf get the same score. Ties between candidates are possible and are broken by distance (nearer first).
- *Limitation, overfitting:* a single deep tree can fit noise in the training trips (see C9). The two directions of a
  two-way road have nearly equal distance, so a tree that splits too finely may latch onto tiny differences that do not carry
  over to other trips.
- *Limitation, instability:* a small change in the training data can change the tree a lot, which is what Random Forest
  averages away.

## B. Training and configuration

**5. Hyperparameters and initial values** (`configs/models/dtree.yaml`; no tuning was done for this record):

| Parameter | Value | Meaning |
|---|---|---|
| `max_depth` | 12 | longest chain of questions; limits how detailed the tree can get |
| `min_samples_leaf` | 50 | every leaf must hold at least 50 training rows, so probabilities rest on enough rows |
| `criterion` | gini | split quality measure (impurity) |
| `max_features` | null | every split considers all features |
| seed | 42 | makes the result repeatable |

Training data: every 3rd observation of 78 trips (trips 00000054 and 00000033 excluded), 558,237 rows
(`obs_stride` 3). The final fit takes 4.2 s.

**6. Loss function / objective.**
The tree is grown greedily. At each node it picks the feature and threshold that most reduce the Gini impurity of the two
child nodes, and it stops when `max_depth` or `min_samples_leaf` would be violated. There is no gradient-based optimisation.

## C. Experimental results

**7. Performance on student_test** (10 trips, 18,631 labelled observations, 195,635 candidate rows; `test` block of the record):

| Metric | Value |
|---|---|
| Top-1 accuracy | 0.8513 |
| Macro F1 | 0.9154 |
| Precision (correct-arc class) | 0.8656 |
| Recall (correct-arc class) | 0.8282 |
| Log loss | 0.0803 |
| Inference time | 0.039 ms / 1,000 rows |
| Candidate miss rate | 0.0001 |

Confusion matrix on candidate rows (negative class first): 174,610 true negatives, 2,395 false positives, 3,201 false
negatives, 15,429 true positives (`results/figures/confusion_dtree__default.png`).

**8. Observation-level Top-1, Macro F1 and log loss.**
Top-1 is measured per GPS observation: take the highest-scoring candidate and check whether it is the true arc. The Decision
Tree is right on 85.1% of the labelled observations. Among the five default runs it is third on Top-1 and macro F1
(Random Forest 0.8557 / 0.9187 and XGBoost 0.8621 / 0.9195 are higher; `results/model_comparison.csv`), but it is very fast:
0.039 ms per 1,000 rows against 0.47 ms for XGBoost, 2.0 ms for Random Forest and 368 ms for KNN. Only Logistic Regression
is faster (0.019 ms). Its log loss (0.0803) is
better than KNN (0.1128) and Logistic Regression (0.0999) but worse than Random Forest (0.0704) and XGBoost (0.0692), as
expected from probabilities that come from a single tree's leaves. The non-ML nearest-candidate baseline reaches Top-1
0.5399 in the same table, so the tree adds about 31 points.

**9. Over- or underfitting?**
*Cross-validation versus test.* CV (5 folds grouped by trip): Top-1 0.8459 ± 0.0427, macro F1 0.9086 ± 0.0227. student_test:
Top-1 0.8513, macro F1 0.9154. These agree within one fold standard deviation.

*Training-fold versus validation-fold scores* (`results/eval/dtree_train_vs_val.csv`, same folds and 3-stride subsample as
the CV run; the validation columns reproduce the per-fold values in the run record):

| Fold | Train Top-1 | Validation Top-1 | Gap |
|---|---|---|---|
| 0 | 0.8716 | 0.8135 | +0.058 |
| 1 | 0.8814 | 0.7960 | +0.085 |
| 2 | 0.8714 | 0.8283 | +0.043 |
| 3 | 0.8555 | 0.9074 | −0.052 |
| 4 | 0.8569 | 0.8841 | −0.027 |
| mean | 0.8674 | 0.8459 | +0.022 |

On average the tree scores about 2 points higher on trips it trained on, so there is mild overfitting. The picture is not
uniform: in folds 3 and 4 the validation score is higher than the training score, because those folds contain easier trips.
The per-fold validation Top-1 varies by about 11 points (0.796 to 0.907), much more than the 2-point average gap.

*Effect of tree depth* (`results/eval/dtree_depth_sweep.csv`, all other settings unchanged, mean over the 5 folds):

| `max_depth` | Train Top-1 | Validation Top-1 | Gap | Train log loss | Validation log loss |
|---|---|---|---|---|---|
| 3 | 0.8290 | 0.8302 | −0.001 | 0.1125 | 0.1132 |
| 6 | 0.8452 | 0.8435 | +0.002 | 0.0889 | 0.0909 |
| 12 (default) | 0.8674 | 0.8459 | +0.021 | 0.0722 | 0.0965 |
| 20 | 0.8725 | 0.8452 | +0.027 | 0.0670 | 0.1182 |
| unlimited | 0.8725 | 0.8451 | +0.027 | 0.0669 | 0.1192 |

Going deeper keeps improving the training score and the training log loss but not the validation Top-1: it peaks at depth 12
(0.8459) and is flat after that, while validation log loss keeps getting worse (0.0965 at depth 12, 0.1192 unlimited). That
is the typical sign of overfitting: the extra splits fit the training trips and make the probabilities overconfident on new
ones. At depth 3 the tree is slightly underfit (validation Top-1 0.8302, 1.6 points lower). So depth 12 is a reasonable
middle, though the validation Top-1 differences between depths 6 and 30 are far smaller than the fold-to-fold spread
(standard deviation about 0.04). The tree never reaches a perfect training score because `min_samples_leaf` is 50 in every
run of the sweep; this setting limits the overfitting a lot.

## Figures
- `results/figures/confusion_dtree__default.png`
- `results/figures/inference_time.png`
