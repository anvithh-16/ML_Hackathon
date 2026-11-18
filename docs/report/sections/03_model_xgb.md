# 3.2 XGBoost  (owner: Member 4)

> Organiser rule: answers must connect the algorithm's behaviour to OUR experiments. Every number must come from a file
> in `results/` (cite the file). Write in your own words. Do not paste numbers you did not produce.

Run record: `results/experiments/xgb__default__<timestamp>.json` · Config: `configs/models/xgb.yaml`

## A. Model understanding
1. **Why is this algorithm suitable for PathSnap?** (boosting; sequential error correction; hist trees; efficiency)
2. **Major strengths and limitations for this task** (relate to our features: distance, heading, two-way roads, coarse/smooth probabilities, speed).

## B. Training and configuration
5. **Important hyperparameters and initial values** (copy from the YAML; explain each in one line).
6. **Loss function / optimisation objective.**

## C. Experimental results (student_test, all observations, labelled rows)
7. **Performance on the student-testing dataset** - table: Top-1, Macro F1, precision, recall, log loss, inference time.
8. **Observation-level Top-1, Macro F1 and Log Loss** - cite the JSON record.
9. **Over- or underfitting?** Evidence: CV mean ± std vs student_test; training-fold vs validation-fold scores; per-fold spread.

## Figures
- `results/figures/confusion_xgb__default.png`

- Optional real training curve: per-iteration train/validation log loss on one fold (only XGBoost produces one).
