"""One JSON file per experiment run (never a shared CSV -> no merge conflicts). Owner: Member 2."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.common.config import load_config, path_of, project_root


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=project_root(),
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def experiments_dir() -> Path:
    d = path_of(load_config(), "results") / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_experiment(record: dict) -> Path:
    rec = dict(record)
    rec.setdefault("timestamp", datetime.now().strftime("%Y%m%d-%H%M%S"))
    rec.setdefault("git_hash", git_hash())
    path = experiments_dir() / f"{rec['model']}__{rec['variant']}__{rec['timestamp']}.json"
    path.write_text(json.dumps(rec, indent=1, default=float))
    return path


def load_experiments(latest_only: bool = True, include_synthetic: bool = False) -> pd.DataFrame:
    recs = []
    for f in sorted(experiments_dir().glob("*.json")):
        r = json.loads(f.read_text())
        r["_file"] = str(f)
        recs.append(r)
    if not recs:
        return pd.DataFrame()
    df = pd.json_normalize(recs)
    if not include_synthetic and "synthetic" in df:
        df = df[~df["synthetic"].fillna(False).astype(bool)]
    if latest_only and len(df):
        df = df.sort_values("timestamp").groupby(["model", "variant"], as_index=False).tail(1)
    return df.reset_index(drop=True)
