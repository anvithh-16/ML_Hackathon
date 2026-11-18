<!-- AI-assisted first draft (Claude Code). Member 4: rewrite in your own words before submission; every number is traced to a file below. -->
# 3.2 XGBoost  (owner: Member 4)

Run record: `results/experiments/xgb__default__20261118-144202.json` · Config: `configs/models/xgb.yaml` ·
Model: `artifacts/models/xgb__default/` (selected and exported to `artifacts/final/`).

Reproduce:

```bash
python -m src.eval.cv_runner --model xgb --variant default --dump-scores
python -m src.eval.train_vs_val --model xgb        # train vs validation, per fold
```

## A. Model understanding

**1. Why XGBoost suits PathSnap.**
Each row is one (GPS observation, candidate arc) pair and the question is "is this the right arc?". The answer depends
on combinations of features rather than on any one of them: a candidate 6 m away is usually right, unless its heading
disagrees by 170° because it is the reverse twin of a two-way road. Gradient boosting builds trees one after another,
and each new tree is fitted to the errors the previous trees still make, so later trees concentrate on these hard cases
(reverse twins, parallel roads, points near junctions). The `hist` tree method bins each feature, which keeps training on
558 237 rows to about 2 s per fit (run record, `train_time_s` = 2.27 s).

**2. Strengths and limitations for this task.**
- *Interactions:* trees split on distance, then heading, then neighbouring-observation consistency without us
  hand-crafting products of features; Logistic Regression cannot do this and scores lower (Top-1 0.843 vs 0.862,
  `results/model_comparison.csv`).
- *Probabilities:* boosting with log loss gives smooth, well-ranked probabilities. XGBoost has the lowest student_test log
  loss of all models (0.0692), which matters because the HMM uses these probabilities as emission scores.
- *Speed:* 0.47 ms per 1 000 candidate rows, about 4× faster than Random Forest (2.02 ms) and ~800× faster than KNN.
- *No scaling needed:* trees only compare thresholds, so distance in metres and heading in degrees can be used as is.
- *Limitations:* it scores each candidate independently, so on its own it can jump between unconnected arcs (classifier-only
  route continuity 0.935); the HMM fixes this. It cannot recover an arc that candidate generation missed (miss rate
  0.0054 %). It fits the training trips more closely than the validation trips (see 9).

## B. Training and configuration

**5. Hyperparameters and initial values** (`configs/models/xgb.yaml`; these defaults are the selected model):

| Parameter | Value | Meaning |
|---|---|---|
| `n_estimators` | 400 | number of boosting rounds (trees) |
| `learning_rate` | 0.1 | how much each new tree's correction is scaled down |
| `max_depth` | 6 | depth limit of each tree (interaction order) |
| `subsample` | 0.8 | share of rows sampled for each tree (adds randomness, reduces overfitting) |
| `colsample_bytree` | 0.8 | share of features sampled for each tree |
| `min_child_weight` | 1 | minimum hessian sum in a leaf (larger = more conservative splits) |
| `reg_lambda` | 1.0 | L2 penalty on leaf weights |
| `tree_method` | hist | histogram-based split finding (fast) |

Common settings for every model: seed 42, feature version v1, every 3rd observation of each training trip, 78 training
trips (two bad-reference trips excluded), 5 group-aware folds by trip. No probability calibration.

**6. Loss function / objective.**
`binary:logistic`: the model outputs a logit, the sigmoid turns it into P(correct arc), and training minimises binary
log loss (cross-entropy). Each boosting round fits a tree to the gradient and hessian of this loss, plus the L2 penalty
`reg_lambda` on leaf weights. Evaluation uses observation-level Top-1 (main metric), candidate-row Macro F1, precision,
recall and log loss.

## C. Experimental results

**7–8. student_test performance** (`results/experiments/xgb__default__20261118-144202.json`; 10 trips,
18 631 labelled observations, 195 635 candidate rows).

| Metric | student_test |
|---|---:|
| Observation-level Top-1 accuracy | **0.8621** |
| Candidate-row Macro F1 | **0.9195** |
| Precision / recall (positive class, threshold 0.5) | 0.8777 / 0.8313 |
| Log loss | **0.0692** |
| Candidate miss rate | 0.0000537 |
| Inference time | 0.468 ms per 1 000 rows |
| Training time (full training set) | 2.27 s |

Confusion matrix (`results/figures/confusion_xgb__default.png`): 15 487 true positives, 3 143 correct arcs missed,
2 158 false alarms, 174 847 true negatives. Missing a correct arc is more common than a false alarm, which explains
recall < precision.

Five-fold CV on the training trips (mean ± std): Top-1 0.8504 ± 0.0436, Macro F1 0.9125 ± 0.0235, log loss 0.0829 ± 0.0225.

After the HMM (`results/hmm/clf_vs_hmm.csv`, mean over the 10 student_test trips): Top-1 rises from 0.880
(classifier-only) to 0.942 (classifier + HMM), and route continuity from 0.935 to 0.979.

**9. Over- or underfitting?** (`results/eval/xgb_train_vs_val.csv`)

| | Train folds (mean) | Validation folds (mean) | Gap |
|---|---:|---:|---:|
| Top-1 | 0.915 | 0.853 | 0.061 |
| Macro F1 | 0.947 | 0.914 | 0.034 |
| Log loss | 0.051 | 0.082 | — |

- There is a moderate train/validation gap: the model fits the trips it has seen better than unseen trips. The gap
  differs a lot between folds (Top-1 gap 0.09–0.11 on folds 0–2, about 0 on folds 3–4), so part of it is which trips
  land in a fold, not only model complexity.
- It is not harmful overfitting: student_test Top-1 (0.862) and Macro F1 (0.920) are at or slightly above the CV means,
  so performance carries over to new trips.
- It is not underfitting: it has the best Top-1 and log loss of all five models.
- Making the model more conservative did not help: Optuna explored depth, regularisation and sampling (section 4) and
  the best configuration was no better on student_test.

## Figures
- `results/figures/confusion_xgb__default.png`
- `results/figures/test_top1_macro_f1.png`, `results/figures/test_logloss.png` (comparison with the other models)
