# 3.2 K-Nearest Neighbours  (owner: Member 2)

Run record: `results/experiments/knn__default__20261118-130751.json` · Config: `configs/models/knn.yaml` · Code: `src/models/knn.py`
All numbers below are read from that record unless another file is named.

## A. Model understanding

**1. Why is this algorithm suitable for PathSnap?**
We turn map matching into a yes/no question for every (GPS point, candidate arc) pair: "is this the arc the vehicle was on?".
Each pair is described by the same fixed list of numbers (`FEATURE_COLUMNS`, for example the distance to the arc and the
heading difference). KNN answers by finding the training pairs whose feature vectors are closest to the new one and checking
how many of them were correct matches. This suits the task because a candidate that is close to the GPS point and points the
same way as the vehicle should look like earlier candidates that turned out to be right. KNN makes no assumption about the
shape of the decision boundary. Because it is distance based, the features must be on comparable scales, so the model
standardises them (`StandardScaler`) inside the same pipeline; the scaling is therefore learned from training rows only.

**2. Strengths and limitations for this task.**
- *Strength:* almost no training time (0.26 s for the final fit), and it can follow irregular patterns in the data.
- *Limitation, cost:* there is no compact model to look up at prediction time. Every new row is compared with all stored
  training rows. The final model stores 558,237 rows and its artifact is about 33 MB (`artifacts/models/knn__default`).
  Measured prediction time is **367.6 ms per 1,000 candidate rows**, against 0.039 ms for the Decision Tree
  (`results/experiments/dtree__default__20261118-130319.json`), roughly 9,500 times slower. Scoring one validation fold took
  36–42 s (`infer_s` per fold in the record).
- *Limitation, coarse probabilities:* the output is a vote share among 25 neighbours, so it takes few distinct values.
  This hurts log loss (see C8).
- *Limitation, equal feature weights:* after scaling every feature counts equally, whether or not it matters. On two-way
  roads the two directions of one road lie at almost the same distance, so only the heading feature separates them, and KNN
  cannot learn that this feature deserves more weight.

## B. Training and configuration

**5. Hyperparameters and initial values** (`configs/models/knn.yaml`; no tuning was done for this record):

| Parameter | Value | Meaning |
|---|---|---|
| `n_neighbors` | 25 | how many nearest training rows vote |
| `weights` | distance | closer neighbours count more |
| `p` | 2 | Euclidean distance (1 would be Manhattan) |
| `algorithm` | auto | scikit-learn picks the search structure |
| scaler | StandardScaler | zero mean, unit variance per feature, fit on training rows only |

The final model is trained on every 3rd observation (`obs_stride` 3) of 78 trips, which gives 558,237 rows. Trips 00000054
and 00000033 are excluded because their reference routes are bad. Seed 42.

**6. Loss function / objective.**
KNN has no loss function and no optimisation step. "Training" only stores the scaled data. The prediction for a row is the
distance-weighted share of its 25 nearest neighbours that were correct matches.

## C. Experimental results

**7. Performance on student_test** (10 trips, 18,631 labelled observations, 195,635 candidate rows; `test` block of the record):

| Metric | Value |
|---|---|
| Top-1 accuracy | 0.8321 |
| Macro F1 | 0.8997 |
| Precision (correct-arc class) | 0.8485 |
| Recall (correct-arc class) | 0.7891 |
| Log loss | 0.1128 |
| Inference time | 367.6 ms / 1,000 rows |
| Candidate miss rate | 0.0001 |

Confusion matrix on candidate rows (negative class first): 174,381 true negatives, 2,624 false positives, 3,929 false
negatives, 14,701 true positives (`results/figures/confusion_knn__default.png`). Precision is higher than recall, so KNN is
slightly cautious: it misses more correct arcs than it wrongly accepts.

**8. Observation-level Top-1, Macro F1 and log loss.**
Top-1 is measured per GPS observation: among that observation's candidates take the one with the highest score and check
whether it is the true arc. KNN is right on 83.2% of the labelled observations. Macro F1 (0.8997) is higher than Top-1
because it also counts the many easy "wrong arc" rows. Log loss (0.1128) is the worst of the five default runs
(Decision Tree 0.0803, Random Forest 0.0704, XGBoost 0.0692, Logistic Regression 0.0999; `results/model_comparison.csv`),
because a vote share of exactly 0 or 1 is punished hard when it is wrong (probabilities are clipped at 1e-6). KNN also has
the lowest Top-1 of the five. Compared with the non-ML nearest-candidate baseline (Top-1 0.5399 in the same table) it is
much better, about 29 points, so the features and the model clearly add information.

**9. Over- or underfitting?**
- Cross-validation (5 folds grouped by trip): Top-1 0.8353 ± 0.0378, macro F1 0.8969 ± 0.0208, log loss 0.1241 ± 0.0283.
  On student_test: Top-1 0.8321, macro F1 0.8997, log loss 0.1128. The two agree to within one fold standard deviation, so
  the model does not just memorise the trips it was trained on.
- Per-fold Top-1 ranges from 0.7925 to 0.8890. This spread is larger than the CV-versus-test difference, so it comes from
  which trips fall in a fold (some are simply harder), not from an unstable model.
- I did not measure training-fold accuracy for KNN. With distance weighting a training row is its own nearest neighbour at
  distance zero, so the training score would be close to perfect and would say nothing about generalisation. The fit is
  therefore judged from the validation folds and student_test only.
- There is no strong sign of underfitting either, but KNN is the weakest of the five on Top-1, and the much simpler
  Decision Tree scores higher.

## Figures
- `results/figures/confusion_knn__default.png`
- `results/figures/inference_time.png` (log scale, shows the KNN cost)
