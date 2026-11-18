<!-- AI-assisted first draft (Claude Code). Member 4: rewrite the summary and answers in your own words and practise saying them aloud. -->
# Viva notes - Member 4

Everyone must be able to explain the WHOLE pipeline; these are the topics you lead on: boosting, TPE/Bayesian optimisation, tuned vs untuned, API flow, where ground truth lives, deployment.

## My 60-second summary of my part
I was the integration coordinator. I set up the repository and the shared contract: one fixed list of 15 features
(`src/common/schema.py`), one config file (`configs/base.yaml`) and one base class for all five models
(`src/models/base.py`), so every model is trained, saved and evaluated the same way on the same columns. My model was
XGBoost; it had the best student_test Top-1 (0.862) and log loss (0.069), so it was selected. I wrote the Optuna
tuning: 40 TPE trials on group-aware CV of the training trips. Tuning did not help on student_test (0.858 vs 0.862), so
we kept the untuned model as we had decided in advance. Finally I own the FastAPI backend: it runs the whole pipeline
on Laptop 1, and the reference route is read only by the server-side scorer after prediction.

## Likely questions and my answers
1. **Q: Boosting vs bagging?**
   A: Bagging (Random Forest) trains many deep trees independently on bootstrap samples and averages them, which mainly
   reduces variance. Boosting (XGBoost) trains shallow trees one after another, each fitted to the remaining errors
   (gradient of the loss) of the ones before, which mainly reduces bias. Our results: XGBoost 0.862 Top-1 vs Random
   Forest 0.856, and XGBoost is about 4× faster at inference (0.47 vs 2.02 ms per 1 000 rows).
2. **Q: What does `learning_rate` do, and how does it interact with `n_estimators`?**
   A: Each tree's contribution is multiplied by it. A smaller rate needs more trees for the same fit but usually
   generalises a bit better. Optuna's best had rate 0.0105 with 500 trees vs our default 0.1 with 400.
3. **Q: What loss does XGBoost minimise here?**
   A: Binary log loss (`binary:logistic`) plus an L2 penalty on leaf weights (`reg_lambda`). Each round uses the
   gradient and second derivative (hessian) of the loss to fit the next tree.
4. **Q: Explain TPE.**
   A: After some trials, Optuna splits the tried configurations into a good group and a bad group by score, fits a
   density to the parameters of each, and picks the next configuration where good/bad density is highest. It focuses
   the budget on promising regions, unlike grid or random search.
5. **Q: How did you make the tuned-vs-untuned comparison fair?**
   A: Same 78 training trips, same 5 group-aware folds, same metric. Trial 0 of the search was the untuned defaults on
   the same stride-6 data. Afterwards both variants were retrained by the same runner on identical partitions and
   compared on student_test (`results/tuning/tuned_vs_untuned.csv`). student_test was never used for tuning.
6. **Q: Tuning improved CV, so why not use the tuned model?**
   A: The CV gain was +0.0026 Macro F1, about a tenth of the fold standard deviation (≈0.023), so it is noise. On
   student_test it was slightly worse on Top-1 and twice as slow. Our rule, fixed before tuning, was student_test Top-1.
7. **Q: Is your model overfitting?**
   A: Train-fold Top-1 0.915 vs validation-fold 0.853 (`results/eval/xgb_train_vs_val.csv`), so it fits seen trips
   better, and the gap depends strongly on the fold. But student_test Top-1 (0.862) is at the CV level, so it
   generalises to new trips.
8. **Q: Why GroupKFold by trip and not random rows?**
   A: Neighbouring GPS points of one trip are almost identical. Random splitting would put near-copies in both training
   and validation and overstate accuracy. Grouping by trip tests on unseen trips, like the evaluator's.
9. **Q: Walk through one `/predict` request.**
   A: Load the trip and project to UTM → candidate arcs within 50 m → 15 features (no labels) → XGBoost probabilities →
   HMM/Viterbi gives one connected route → convert to lat/lon → scorer compares with `.route` if present → JSON with
   route, metrics and timings. About 0.18 s for trip 00000007; the HMM is the slowest stage.
10. **Q: Where does the ground truth live, and how do you know it doesn't leak?**
    A: Only `api/scorer.py` reads `.route`, after prediction. A test fails if `load_route` appears in `api/pipeline.py`,
    the response schema has no label field, and the UI only gets aggregate metrics.
11. **Q: What happens if there is no trained model?**
    A: The API starts in `distance-only` mode: the HMM uses candidate distance alone. `/health` shows the mode.
12. **Q: Laptop 2 can't reach the API — what do you check?**
    A: API started with `--host 0.0.0.0`; same network (phone hotspot if campus Wi-Fi isolates clients); macOS firewall
    allows Python; correct IP from `ipconfig getifaddr en0`; port 8000 not already in use.
13. **Q: Why is XGBoost good for the HMM in particular?**
    A: The HMM uses the probabilities as emission scores, so their quality matters, not just the ranking. XGBoost had
    the lowest log loss (0.0692). With the HMM, mean per-trip Top-1 went from 0.880 to 0.942.

## Numbers I must know (copy from results/ files, with file names)
- XGBoost student_test: Top-1 **0.862**, Macro F1 0.919, log loss 0.069, precision 0.878, recall 0.831 —
  `results/experiments/xgb__default__20261118-144202.json`
- XGBoost CV: Top-1 0.850 ± 0.044, Macro F1 0.913 ± 0.024 — same file
- Other models' Top-1: Random Forest 0.856, Decision Tree 0.851, LogReg 0.843, KNN 0.832, nearest-candidate baseline
  0.540 — `results/model_comparison.csv`
- Inference: XGBoost 0.47 ms / 1 000 rows, Random Forest 2.02, KNN 368 — `results/model_comparison.csv`
- Optuna: 40 trials, best CV Macro F1 0.9157 vs untuned 0.9134 (trial 0) — `results/tuning/xgb_best_params.json`
- Tuned vs untuned student_test Top-1: 0.858 vs 0.862 → untuned kept — `results/tuning/tuned_vs_untuned.csv`,
  `artifacts/final/manifest.json`
- Train vs validation Top-1: 0.915 vs 0.853 — `results/eval/xgb_train_vs_val.csv`
- With HMM (mean over 10 trips): Top-1 0.880 → 0.942 — `results/hmm/clf_vs_hmm.csv`
- API timing, trip 00000007: 0.18 s total, HMM 0.10 s — `results/ui/ui_demo_00000007.json`
- Data: 80 train trips (78 used), 10 student_test trips, 18 631 labelled student_test observations
