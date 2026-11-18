<!-- AI-assisted first draft (Claude Code). Member 4: rewrite in your own words before submission; every number is traced to a file below. -->
# 4. Bayesian hyperparameter optimisation  (owner: Member 4)

Files: `results/tuning/xgb_trials.csv` (one row per trial), `results/tuning/xgb_best_params.json`,
`results/tuning/xgb_history.png`, `results/tuning/tuned_vs_untuned.csv`. Code: `src/tuning/optuna_search.py`.

```bash
python -m src.tuning.optuna_search --model xgb --n-trials 40 --timeout 1800
python -m src.eval.cv_runner --model xgb --variant tuned --params results/tuning/xgb_best_params.json --dump-scores
```

We tuned XGBoost because it was the best model in the comparison (section 3.1). The method is Optuna's TPE sampler
(Tree-structured Parzen Estimator, seed 42): after a few trials it splits the tried configurations into "good" and
"bad" groups by score, models the parameter distribution of each group, and proposes the next configuration where the
ratio good/bad is highest. Unlike grid or random search, it spends more trials near promising regions.

**1–2. Hyperparameters searched and their ranges** (`configs/models/xgb.yaml: search_space`)

| Parameter | Range | Distribution | Default (untuned) |
|---|---|---|---|
| `n_estimators` | 100 – 1000 | integer, step 50 | 400 |
| `learning_rate` | 0.01 – 0.3 | log-uniform | 0.1 |
| `max_depth` | 3 – 10 | integer | 6 |
| `min_child_weight` | 1 – 50 | log-uniform | 1 |
| `subsample` | 0.5 – 1.0 | uniform | 0.8 |
| `colsample_bytree` | 0.5 – 1.0 | uniform | 0.8 |
| `reg_lambda` | 0.001 – 10 | log-uniform | 1.0 |

Learning rate, `min_child_weight` and `reg_lambda` are searched on a log scale because their effect is multiplicative
(0.01 → 0.02 matters as much as 0.1 → 0.2).

**3. Objective.** Mean Macro F1 over the same 5 GroupKFold folds of the 78 training trips (folds by trip, so
observations from one trip never appear in both training and validation), using every 6th observation
(`tuning.obs_stride` in `configs/base.yaml`) to keep each trial fast. student_test and evaluator data were never used
for tuning. Trial 0 was the untuned defaults, so the starting point was measured on exactly the same data and folds.

**4. Trials completed.** 40 of 40 (all `COMPLETE`, `results/tuning/xgb_trials.csv`), about 375 s of cross-validation
in total. Trial scores ranged from 0.9016 to 0.9157.

**5. Best values** (`results/tuning/xgb_best_params.json`, found at trial 32):
`n_estimators` 500, `learning_rate` 0.0105, `max_depth` 8, `subsample` 0.636, `colsample_bytree` 0.826,
`min_child_weight` 1.93, `reg_lambda` 2.45. Objective 0.9157 vs 0.9134 for the untuned trial 0.
The search moved to a much smaller learning rate with deeper trees and more regularisation, i.e. many small corrections
instead of fewer large ones.

**6. Did tuning help? No.** Both variants were then retrained by the same runner, on the same folds and partitions
(every 3rd observation, as for all models), and compared (`results/tuning/tuned_vs_untuned.csv`):

| | Untuned (`xgb__default`) | Tuned (`xgb__tuned`) |
|---|---:|---:|
| CV Macro F1 (mean ± std) | 0.9125 ± 0.0235 | 0.9151 ± 0.0224 |
| CV Top-1 | 0.8504 | 0.8517 |
| student_test Top-1 | **0.8621** | 0.8580 |
| student_test Macro F1 | 0.9195 | 0.9194 |
| student_test log loss | **0.0692** | 0.0693 |
| Inference (ms / 1 000 rows) | **0.47** | 0.92 |
| Training time (s) | **2.27** | 3.96 |

- The CV gain (+0.0026 Macro F1) is about a tenth of the fold-to-fold standard deviation (~0.023), so it is within noise.
- On student_test the tuned model is slightly *worse* on Top-1 (0.858 vs 0.862), equal on Macro F1 and log loss, and twice
  as slow at inference.
- Our pre-declared rule was to keep the better model by student_test Top-1, so we kept the **untuned** model
  (`artifacts/final/manifest.json`, `selection_reason`).

Why tuning gave so little: the defaults were already in a good region of the search space, and with 15 features and over
half a million rows, XGBoost's accuracy is limited more by the information in the features (ambiguous reverse twins and
parallel roads) than by its hyperparameters. The HMM then corrects many of the remaining per-point errors.
(The untuned objective is 0.9134 during tuning but 0.9125 in the runner because tuning used every 6th observation and
the runner every 3rd.)

Figure: `results/tuning/xgb_history.png` — CV Macro F1 per trial and best-so-far curve.
