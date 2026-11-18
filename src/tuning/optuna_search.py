"""Bayesian hyperparameter optimisation (Optuna TPE) with group-aware CV. Owner: Member 4.

python -m src.tuning.optuna_search --model xgb --synthetic --n-trials 5      # smoke test (MOCK data)
python -m src.tuning.optuna_search --model <selected> --n-trials 40
Objective: mean 5-fold GroupKFold Macro F1 on the TRAIN partition (tuning.obs_stride rows),
computed by Member 2's cross_validate(). The untuned baseline is evaluated with the SAME stride
and folds (trial 0 is enqueued with the default params) so the comparison is fair.
"""
from __future__ import annotations

import argparse
import json
import time

import optuna
import pandas as pd

from src.common.config import load_config, load_model_config, path_of
from src.common.logging_utils import get_logger
from src.eval.cv_runner import cross_validate, load_folds, load_partition_features
from src.models.registry import get_model_class

log = get_logger("optuna")


def suggest(trial: optuna.Trial, space: dict) -> dict:
    p = {}
    for name, s in space.items():
        t = s["type"]
        if t == "int":
            p[name] = trial.suggest_int(name, int(s["low"]), int(s["high"]), step=int(s.get("step", 1)),
                                        log=bool(s.get("log", False)) and int(s.get("step", 1)) == 1)
        elif t == "float":
            p[name] = trial.suggest_float(name, float(s["low"]), float(s["high"]), log=bool(s.get("log", False)))
        elif t == "categorical":
            p[name] = trial.suggest_categorical(name, list(s["choices"]))
        else:
            raise ValueError(f"unknown search-space type {t!r} for {name}")
    return p


def tune(model: str, n_trials: int, synthetic: bool = False, timeout: int | None = None) -> dict:
    cfg = load_config()
    mcfg = load_model_config(model)
    space = mcfg.get("search_space") or {}
    if not space:
        raise SystemExit(f"configs/models/{model}.yaml has no search_space")
    defaults = dict(mcfg.get("params", {}))
    cls = get_model_class(model)
    if synthetic:
        from tests.fixtures.synthetic import make_synthetic, synthetic_folds
        df = make_synthetic(n_trips=20, seed=cfg["seed"])
        folds = synthetic_folds(df)
    else:
        df = load_partition_features("train", cfg, cfg["tuning"]["obs_stride"])
        folds = load_folds()
    log.info("tuning %s on %d rows / %d trips", model, len(df), df.trip_id.nunique())

    def objective(trial):
        params = dict(defaults)
        params.update(suggest(trial, space))
        t0 = time.perf_counter()
        res = cross_validate(cls, params, df, folds, cfg["seed"])
        trial.set_user_attr("cv_top1_mean", res["top1_mean"])
        trial.set_user_attr("cv_logloss_mean", res["logloss_mean"])
        trial.set_user_attr("cv_macro_f1_std", res["macro_f1_std"])
        trial.set_user_attr("duration_s", time.perf_counter() - t0)
        return res["macro_f1_mean"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
                                study_name=f"{model}_tuning")
    # trial 0 = the untuned defaults (only for keys in the search space)
    study.enqueue_trial({k: v if v is not None else "null" for k, v in defaults.items() if k in space})
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    out = path_of(cfg, "results") / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{model}{'_synthetic' if synthetic else ''}"
    trials = study.trials_dataframe()
    trials.to_csv(out / f"{tag}_trials.csv", index=False)
    best = dict(defaults)
    best.update(study.best_params)
    (out / f"{tag}_best_params.json").write_text(json.dumps(
        {"model": model, "objective": "cv_macro_f1_mean", "best_value": study.best_value,
         "untuned_value": study.trials[0].value, "n_trials": len(study.trials), "synthetic": synthetic,
         "params": best}, indent=1, default=str))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        vals = trials["value"].to_numpy()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(range(len(vals)), vals, "o", ms=4, label="trial")
        ax.plot(range(len(vals)), pd.Series(vals).cummax(), "-", label="best so far")
        ax.set_xlabel("trial")
        ax.set_ylabel("CV Macro F1 (train, GroupKFold)")
        ax.set_title(f"Optuna TPE history - {model}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{tag}_history.png", dpi=150)
    except Exception as e:
        log.warning("plot failed: %s", e)
    log.info("best %.4f (untuned %.4f) params %s", study.best_value, study.trials[0].value, study.best_params)
    return {"best_value": study.best_value, "untuned_value": study.trials[0].value, "params": best}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["logreg", "knn", "dtree", "rforest", "xgb"])
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=None, help="seconds")
    ap.add_argument("--synthetic", action="store_true", help="MOCK data smoke test")
    a = ap.parse_args()
    n = a.n_trials or load_config()["tuning"]["n_trials"]
    r = tune(a.model, n, a.synthetic, a.timeout)
    print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main()
