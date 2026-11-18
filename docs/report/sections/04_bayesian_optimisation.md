# 4. Bayesian hyperparameter optimisation  (owner: Member 4)

> Organiser rule: answers must connect the algorithm's behaviour to OUR experiments. Every number must come from a file
> in `results/` (cite the file). Write in your own words. Do not paste numbers you did not produce.

Files: `results/tuning/<model>_trials.csv`, `<model>_best_params.json`, `<model>_history.png`, `tuned_vs_untuned.csv`.

1. Hyperparameters included in the search.
2. Ranges / distributions (from `configs/models/<model>.yaml: search_space`).
3. Objective: mean 5-fold GroupKFold Macro F1 on training trips (stride 6) - why.
4. Number of trials actually completed.
5. Best values.
6. Did the tuned model beat the untuned one? Table on identical partitions: CV Macro F1, student_test Top-1, student_test Macro F1, log loss, inference time.
