"""Copy the selected model + configs into artifacts/final/ (what the API loads). Owner: Member 4.

python -m src.export.export_artifacts --model xgb --variant tuned --reason "best student_test Top-1"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from datetime import datetime

from src.common.config import load_config, path_of, project_root
from src.common.schema import FEATURE_VERSION
from src.eval.experiment_log import git_hash, load_experiments


def export(model: str, variant: str, reason: str) -> dict:
    cfg = load_config()
    src = path_of(cfg, "artifacts") / "models" / f"{model}__{variant}"
    if not (src / "model.joblib").exists():
        raise SystemExit(f"{src} has no model.joblib - train it with src.eval.cv_runner first")
    dst = project_root() / cfg["api"]["final_artifacts"]
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("model.joblib", "config.json"):
        shutil.copy2(src / f, dst / f)
    shutil.copy2(project_root() / "configs" / "hmm.yaml", dst / "hmm.yaml")
    shutil.copy2(project_root() / "configs" / "base.yaml", dst / "base.yaml")
    exps = load_experiments()
    rec = exps[(exps.model == model) & (exps.variant == variant)]
    manifest = {"model": model, "variant": variant, "feature_version": FEATURE_VERSION, "git_hash": git_hash(),
                "created_at": datetime.now().isoformat(timespec="seconds"), "selection_reason": reason,
                "source_artifact": str(src.relative_to(project_root())),
                "source_record": str(Path(rec["_file"].iloc[0]).relative_to(project_root())) if len(rec) else None}
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="tuned")
    ap.add_argument("--reason", required=True, help="why this model/variant was selected (goes in manifest)")
    a = ap.parse_args()
    print(json.dumps(export(a.model, a.variant, a.reason), indent=1))


if __name__ == "__main__":
    main()
