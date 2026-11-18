# 3.2 Random Forest (owner: Member 3)

Run record: `results/experiments/rforest__default__20261118-131637.json`  
Config: `configs/models/rforest.yaml`

Reproduce the experiment and regenerate the model artifact:

```bash
python -m src.eval.cv_runner --model rforest --variant default --dump-scores
```

`artifacts/models/rforest__default/model.joblib` is approximately 55 MB, above the repository's 50 MB artifact limit. It is intentionally not committed; regenerate it with the command above.

## A. Model understanding

**1. Why Random Forest is suitable for PathSnap.**

PathSnap scores candidate road arcs for each GPS observation. Random Forest is an ensemble of decision trees trained on bootstrap samples and feature subsets. Averaging trees reduces the variance of an individual tree and models non-linear decision boundaries and feature interactions. This is useful because distance, heading difference, neighbouring-observation consistency, speed and road direction can jointly influence whether a candidate arc is correct.

**2. Strengths and limitations.**

- Learns non-linear thresholds and interactions, unlike a linear model such as Logistic Regression.
- Is robust to noisy individual features and provides candidate scores that can be used as HMM emissions.
- Is larger and more computationally expensive than Logistic Regression.
- Scores candidates independently; by itself it does not guarantee that consecutive predictions form a connected route. HMM/Viterbi adds sequence consistency.
- Candidate generation remains a hard boundary: if the true arc is absent from an observation's candidate set, the classifier cannot select it.

## B. Training and configuration

**5. Main hyperparameters** (from `configs/models/rforest.yaml`):

- `n_estimators = 300` — number of trees.
- `max_depth = 20` — maximum tree depth, limiting complexity.
- `min_samples_leaf = 10` — minimum rows per leaf, smoothing leaf predictions.
- `max_features = sqrt` — each split considers the square root of available features.
- `n_jobs = -1` — use all available CPU cores.
- `calibration = none` — no additional probability calibration.

The experiment used seed 42, feature version v1, observation stride 3, and five group-aware folds by trip. Training used 78 trips and 558,237 candidate rows. The student-test partition contains 10 trips.

**6. Objective.**

The Random Forest is trained as a binary candidate classifier (correct arc vs. incorrect candidate). Individual trees learn split rules that reduce class impurity; the ensemble combines tree predictions. Evaluation uses observation-level Top-1 and candidate-row Macro F1, precision, recall and log loss.

## C. Experimental results

**7–8. Student-test performance** (`results/experiments/rforest__default__20261118-131637.json`).

The student-test evaluation contains 18,631 labelled observations and 195,635 candidate rows.

| Metric | Student-test result |
|---|---:|
| Observation-level Top-1 accuracy | **0.8557** |
| Candidate-row Macro F1 | **0.9187** |
| Candidate-row precision | 0.8852 |
| Candidate-row recall | 0.8217 |
| Log loss | 0.0704 |
| Candidate miss rate | 0.0000537 (0.00537%) |
| Inference time | 2.018 ms per 1,000 candidate rows |
| Training time | 27.69 s |

Top-1 measures whether the highest-scored candidate for an observation is the labelled road arc. Macro F1, precision and recall are candidate-row classification metrics at the configured decision threshold; they are not the same as observation-level Top-1.

**Five-fold cross-validation (mean ± standard deviation):**

| Metric | CV mean ± std |
|---|---:|
| Top-1 accuracy | 0.8501 ± 0.0414 |
| Macro F1 | 0.9129 ± 0.0226 |
| Log loss | 0.0824 ± 0.0181 |

Fold Top-1 ranged from 0.8014 to 0.9105, and fold Macro F1 ranged from 0.8890 to 0.9453. Student-test Top-1 (0.8557) and Macro F1 (0.9187) are close to the CV means, so these results do not show an obvious train/test generalisation gap. The fold spread indicates that trip difficulty varies.

**9. Interpretation.**

The model performs well on this candidate-ranking task, with 85.57% observation-level Top-1 and 0.0704 log loss on student_test. Its low candidate miss rate is a property of candidate generation, not a capability of Random Forest. Remaining errors can include ambiguous parallel or reverse-direction roads, difficult GPS observations, and cases where the candidate set does not contain the correct arc.

## Figures

- `results/figures/confusion_rforest__default.png` — include if generated and present in the current results directory.
