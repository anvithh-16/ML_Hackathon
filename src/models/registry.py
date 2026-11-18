"""Model name -> class, imported lazily so a missing teammate module never breaks the others.
Owner: Member 4."""
from __future__ import annotations

import importlib

MODEL_REGISTRY = {
    "logreg": "src.models.logreg:LogRegModel",    # Member 1
    "knn": "src.models.knn:KNNModel",             # Member 2
    "dtree": "src.models.dtree:DTreeModel",       # Member 2
    "rforest": "src.models.rforest:RFModel",      # Member 3
    "xgb": "src.models.xgb:XGBModel",             # Member 4
}


def get_model_class(name: str):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_REGISTRY)}")
    mod, cls = MODEL_REGISTRY[name].split(":")
    return getattr(importlib.import_module(mod), cls)


def available_models() -> list[str]:
    """Models whose module can currently be imported (others are still on a branch)."""
    ok = []
    for name in MODEL_REGISTRY:
        try:
            get_model_class(name)
            ok.append(name)
        except (ImportError, AttributeError):
            pass
    return ok
