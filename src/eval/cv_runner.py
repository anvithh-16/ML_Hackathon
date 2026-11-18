"""THE training + evaluation entry point for all five models. Owner: Member 2.

python -m src.eval.cv_runner --model knn --variant smoke --synthetic        # before features exist
python -m src.eval.cv_runner --model knn --variant default --dump-scores    # real run

Protocol (identical for every model):
  * training rows: train partition, minus exclude_trips_train, labelled observations only,
    every `training.obs_stride`-th observation of each trip;
  * 5-fold GroupKFold by trip from configs/folds.json -> per-fold macro F1 / Top-1 / log loss;
  * refit on all training rows -> artifacts/models/<model>__<variant>/;
  * score ALL student_test rows (no stride); metrics on labelled rows;
  * one JSON record in results/experiments/; optional score parquet for the HMM.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config, load_model_config, path_of, project_root
from src.common.logging_utils import get_logger
from src.common.schema import FEATURE_COLUMNS, FEATURE_VERSION, SCORE_COLUMNS
from src.eval.experiment_log import log_experiment
from src.eval.metrics import evaluate_scores
from src.models.registry import get_model_class

log = get_logger("cv_runner")
FEAT_READ = ["trip_id", "obs_idx", "arc_id"] + FEATURE_COLUMNS + ["y", "labelled", "label_arc_id"]


def load_folds(path: str | Path | None = None) -> dict:
    p = Path(path) if path else project_root() / "configs" / "folds.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run `python -m src.data.make_folds` (Member 1)")
    return json.loads(p.read_text())


def load_partition_features(partition: str, cfg: dict, stride: int | None = None,
                            labelled_only: bool = True) -> pd.DataFrame:
    f = path_of(cfg, "processed") / partition / "features.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing - run `python -m src.features.build_features --partition {partition}`")
    df = pd.read_parquet(f, columns=FEAT_READ)
    df["trip_id"] = df["trip_id"].astype(str)
    if partition == "train":
        df = df[~df.trip_id.isin(cfg.get("exclude_trips_train", []))]
    if labelled_only:
        df = df[df.labelled]
    if stride and stride > 1:
        df = df[df.obs_idx % stride == 0]
    return df.reset_index(drop=True)


def cross_validate(model_cls, params: dict, df: pd.DataFrame, folds: dict, seed: int = 42,
                   calibration: str = "none") -> dict:
    """Group-aware CV. df = labelled training rows. Returns per-fold metrics, means and OOF scores."""
    fold_of = df.trip_id.map(folds["folds"])
    if fold_of.isna().any():
        missing = sorted(df.trip_id[fold_of.isna()].unique())[:5]
        raise ValueError(f"trips without a fold in folds.json: {missing}")
    fold_of = fold_of.astype(int).to_numpy()
    oof = np.full(len(df), np.nan)
    per_fold = []
    X = df[FEATURE_COLUMNS]
    y = df["y"].to_numpy()
    for k in range(folds["n_splits"]):
        tr, va = fold_of != k, fold_of == k
        if va.sum() == 0:
            continue
        assert not set(df.trip_id[tr]) & set(df.trip_id[va]), "trip leakage across folds"
        m = model_cls(params, seed, calibration).fit(X[tr], y[tr], groups=df.trip_id[tr].to_numpy())
        t0 = time.perf_counter()
        p = m.predict_proba(X[va])
        infer = time.perf_counter() - t0
        oof[va] = p
        met = evaluate_scores(df[va], p)
        met.update({"fold": k, "train_time_s": m.train_time_s, "infer_s": infer,
                    "n_val_trips": int(df.trip_id[va].nunique())})
        met.pop("confusion", None)
        per_fold.append(met)
        log.info("fold %d: macro_f1=%.4f top1=%.4f logloss=%.4f", k, met["macro_f1"], met["top1"], met["logloss"])
    summ = {}
    for key in ("macro_f1", "top1", "logloss"):
        vals = np.array([f[key] for f in per_fold])
        summ[f"{key}_mean"], summ[f"{key}_std"] = float(vals.mean()), float(vals.std(ddof=0))
    return {"per_fold": per_fold, **summ, "oof": oof}


def run(model_name: str, variant: str = "default", params_override: dict | None = None,
        dump_scores: bool = False, skip_cv: bool = False, synthetic: bool = False,
        calibration: str | None = None, stride: int | None = None) -> dict:
    cfg = load_config()
    mcfg = load_model_config(model_name)
    params = dict(mcfg.get("params", {}))
    params.update(params_override or {})
    calibration = calibration or mcfg.get("calibration", "none")
    seed = cfg["seed"]
    model_cls = get_model_class(model_name)
    stride = stride or cfg["training"]["obs_stride"]

    if synthetic:
        from tests.fixtures.synthetic import make_synthetic, synthetic_folds
        train = make_synthetic(n_trips=20, seed=seed)
        test_all = make_synthetic(n_trips=5, seed=seed + 1)
        folds = synthetic_folds(train)
        stride = 1
    else:
        train = load_partition_features("train", cfg, stride)
        test_all = load_partition_features("student_test", cfg, None, labelled_only=False)
        folds = load_folds()
    log.info("%s/%s: %d training rows from %d trips (stride %d)", model_name, variant, len(train),
             train.trip_id.nunique(), stride)

    rec = {"model": model_name, "variant": variant, "family": model_cls.family, "feature_version": FEATURE_VERSION,
           "seed": seed, "obs_stride": stride, "params": params, "calibration": calibration,
           "synthetic": bool(synthetic), "n_train_rows": len(train), "n_train_trips": int(train.trip_id.nunique())}

    out_scores = path_of(cfg, "results") / "scores" / f"{model_name}__{variant}"
    if not skip_cv:
        cv = cross_validate(model_cls, params, train, folds, seed, calibration)
        rec["cv"] = {k: v for k, v in cv.items() if k != "oof"}
        if dump_scores and not synthetic:
            out_scores.mkdir(parents=True, exist_ok=True)
            s = train[["trip_id", "obs_idx", "arc_id"]].copy()
            s["score"] = cv["oof"]
            s[SCORE_COLUMNS].to_parquet(out_scores / "oof_train.parquet", index=False)

    model = model_cls(params, seed, calibration).fit(train[FEATURE_COLUMNS], train["y"].to_numpy(),
                                                     groups=train.trip_id.to_numpy())
    art = path_of(cfg, "artifacts") / "models" / (f"{model_name}__{variant}" + ("__synthetic" if synthetic else ""))
    model.save(art)
    rec["train_time_s"] = model.train_time_s
    rec["artifact_path"] = str(art.relative_to(project_root()))

    t0 = time.perf_counter()
    p = model.predict_proba(test_all[FEATURE_COLUMNS])
    infer = time.perf_counter() - t0
    rec["infer_ms_per_1k"] = 1000 * infer / max(len(test_all), 1) * 1000
    met = evaluate_scores(test_all, p)
    rec["test"] = met
    rec["test_partition"] = "synthetic" if synthetic else "student_test"
    if dump_scores and not synthetic:
        out_scores.mkdir(parents=True, exist_ok=True)
        s = test_all[["trip_id", "obs_idx", "arc_id"]].copy()
        s["score"] = p
        s[SCORE_COLUMNS].to_parquet(out_scores / "student_test.parquet", index=False)
    path = log_experiment(rec)
    log.info("test: top1=%.4f macro_f1=%.4f logloss=%.4f | record %s", met["top1"], met["macro_f1"], met["logloss"], path)
    rec["_record_path"] = str(path)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="Train + evaluate one model with the shared protocol")
    ap.add_argument("--model", required=True, choices=["logreg", "knn", "dtree", "rforest", "xgb"])
    ap.add_argument("--variant", default="default")
    ap.add_argument("--params", help="JSON file with parameter overrides (e.g. Optuna best params)")
    ap.add_argument("--calibration", choices=["none", "sigmoid", "isotonic"])
    ap.add_argument("--stride", type=int, help="override training.obs_stride (record it in the variant name)")
    ap.add_argument("--dump-scores", action="store_true")
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="MOCK data; results are not reportable")
    a = ap.parse_args()
    override = json.loads(Path(a.params).read_text()) if a.params else None
    if override and "params" in override:
        override = override["params"]
    rec = run(a.model, a.variant, override, a.dump_scores, a.skip_cv, a.synthetic, a.calibration, a.stride)
    cv = rec.get("cv", {})
    print(json.dumps({"model": rec["model"], "variant": rec["variant"], "synthetic": rec["synthetic"],
                      "cv_macro_f1": cv.get("macro_f1_mean"), "cv_top1": cv.get("top1_mean"),
                      "test_top1": rec["test"]["top1"], "test_macro_f1": rec["test"]["macro_f1"],
                      "test_logloss": rec["test"]["logloss"], "train_time_s": rec["train_time_s"],
                      "infer_ms_per_1k": rec["infer_ms_per_1k"], "record": rec["_record_path"]}, indent=1))


if __name__ == "__main__":
    main()
