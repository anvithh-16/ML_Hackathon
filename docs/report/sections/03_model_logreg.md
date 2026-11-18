# 3.2 Logistic Regression  (owner: Member 1)

Run records: `results/experiments/logreg__default__20261118-130204.json` (main run) and
`results/experiments/logreg__balanced__20261118-130216.json` (class-weight variant).
Config: `configs/models/logreg.yaml`; variant overrides: `configs/variants/logreg_balanced.json`.
Artifacts: `artifacts/models/logreg__default/`, `artifacts/models/logreg__balanced/`.
Reproduce: `python -m src.eval.cv_runner --model logreg --variant default --dump-scores`.

## A. Model understanding

**1. Why it suits PathSnap.** Each row is one (GPS observation, candidate arc) pair and the question is
"is this the correct arc?". Logistic Regression gives a calibrated-looking probability per candidate, which is
exactly what the HMM emission step consumes, and it is the simplest model that can combine distance, heading and
neighbour-consistency evidence. It is our linear baseline: anything the tree models gain over it is gain from
non-linear interactions. Features have very different units (metres, degrees, counts), so a `StandardScaler`
sits inside the pipeline; the shared feature table stays unscaled for the other models.

**2. Strengths and limitations on our features.**
Standardised coefficients of the default model (from the saved artifact) show what it learned:

| feature | coef | reading |
|---|---|---|
| dist_rank | −4.48 | nearer-ranked candidates are much more likely to be correct |
| dist_minus_min | −3.14 | being metres behind the nearest candidate is penalised |
| heading_diff_deg | −1.68 | separates a two-way road from its reverse twin |
| next/prev_obs_dist_same_arc | −0.91 / −0.70 | neighbouring GPS points also close to the arc → more likely |
| dist_m | +0.70 | small correction once rank and gap are known |
| speed_mps, dt_prev_s | ≈ 0 | speed/timing barely used linearly |

Strengths: tiny (8 KB artifact), trains in ≈ 0.3 s on 558 k rows, inference ≈ 0.02 ms per 1 000 rows,
smooth probabilities (best log loss of our runs so far, 0.100 on student_test). Limitations: one linear boundary
cannot express interactions such as "heading matters only when the vehicle is moving" or "distance matters more
on dense networks"; heading is a wrap-around angle that a linear term handles crudely; speed is only useful in
combination with other features, so its weight collapses to ≈ 0.

## B. Training and configuration

**5. Hyperparameters (defaults from the YAML).**
- `C = 1.0` — inverse L2 regularisation strength; smaller = stronger shrinkage of coefficients.
- `solver = lbfgs`, `max_iter = 2000` — quasi-Newton optimiser; 2000 iterations so it converges on scaled data
  (no ConvergenceWarning was raised).
- `class_weight = null` — rows weighted equally (≈ 1 positive : 9 negatives). The `balanced` variant reweights
  classes inversely to their frequency.
- Shared protocol: seed 42, training stride 3 (every 3rd observation per trip), 5 group-aware folds by trip,
  trips 00000054 and 00000033 excluded → 78 trips, 558 237 rows.
- Optuna search space (for tuning, §4): `C` log-uniform [1e-3, 100], `class_weight` ∈ {null, balanced}.

**6. Objective.** Minimise the regularised binary cross-entropy (log loss)
`−Σ [y log p + (1−y) log(1−p)] + ‖w‖² / (2C)`, with `p = σ(wᵀx + b)` on standardised features.

## C. Experimental results

**7–8. Student-test performance** (10 trips, 18 631 labelled observations, 195 635 candidate rows).
Top-1 = the highest-probability candidate of an observation is the true arc. Macro F1 / precision / recall
are at the row level with threshold 0.5.

| variant | Top-1 | Macro F1 | Precision (pos) | Recall (pos) | Log loss | Train time | Infer ms/1k rows |
|---|---|---|---|---|---|---|---|
| default | **0.843** | **0.897** | 0.819 | 0.807 | **0.100** | 0.29 s | 0.018 |
| balanced | 0.844 | 0.815 | 0.531 | 0.940 | 0.194 | 0.22 s | 0.014 |
| nearest-candidate baseline | 0.540 | – | – | – | – | – | – |

Baseline = pick the candidate with smallest `dist_m` (computed from `data/processed/student_test/features.parquet`);
on train it scores 0.559. Logistic Regression adds ≈ 30 points of Top-1, mostly by using heading to pick the
right direction of two-way roads.

`balanced` leaves ranking (Top-1) unchanged — reweighting shifts the intercept, not the ordering — but moves
the 0.5 threshold so recall rises to 0.94 at the cost of precision, lowering Macro F1 and doubling log loss.
**Default is kept** (its probabilities are better for the HMM).

**9. Over- or underfitting?**

| | Top-1 | Macro F1 | Log loss |
|---|---|---|---|
| 5-fold CV (mean ± std) | 0.843 ± 0.039 | 0.891 ± 0.022 | 0.115 ± 0.018 |
| student_test | 0.843 | 0.897 | 0.100 |

CV and student_test agree to within one standard deviation, so there is no sign of overfitting — expected for a
16-parameter linear model trained on 558 k rows. The per-fold spread (Top-1 0.80–0.90) comes from trip
difficulty, not instability. If anything the model **underfits**: it cannot use feature interactions, which is
the gap the tree ensembles (§3.4–3.5) are expected to close.

## Figures
- `results/figures/confusion_logreg__default.png` — produced by M2's `src.eval.compare_models` once `feat/compare` is merged.
