"""Write configs/folds.json: train trip -> fold 0..4 (group-aware CV). Owner: Member 1.

Deterministic (seed 42). Trips are shuffled, then sorted by size and dealt round-robin so
folds have similar numbers of observations. Excluded trips keep a fold but are listed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common.config import load_config, path_of, project_root
from src.data.loader import list_trips


def make_folds(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    root = path_of(cfg, "data_root")
    tids = list_trips(root, "train")
    sizes = []
    for t in tids:
        f = root / "train" / t / f"{t}.track"
        with open(f, "rb") as fh:
            sizes.append(sum(1 for _ in fh))
    rng = np.random.default_rng(cfg["seed"])
    df = pd.DataFrame({"trip_id": tids, "n": sizes}).iloc[rng.permutation(len(tids))]
    df = df.sort_values("n", ascending=False, kind="stable").reset_index(drop=True)
    k = cfg["training"]["n_splits"]
    df["fold"] = [i % k for i in range(len(df))]
    return {"seed": cfg["seed"], "n_splits": k,
            "folds": {r.trip_id: int(r.fold) for r in df.sort_values("trip_id").itertuples()},
            "excluded": list(cfg.get("exclude_trips_train", []))}


def main() -> None:
    folds = make_folds()
    out = project_root() / "configs" / "folds.json"
    out.write_text(json.dumps(folds, indent=1))
    counts = pd.Series(folds["folds"]).value_counts().sort_index()
    print(f"wrote {out}; trips per fold: {counts.to_dict()}")


if __name__ == "__main__":
    main()
