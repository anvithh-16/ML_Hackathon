"""XGBoost candidate classifier. Owner: Member 4.

Gradient-boosted trees (binary:logistic). No scaling needed. On macOS, xgboost needs the
OpenMP runtime: `brew install libomp` if import fails.
"""
from xgboost import XGBClassifier

from src.models.base import BaseCandidateModel


class XGBModel(BaseCandidateModel):
    name = "xgb"
    family = "boosting"
    needs_scaling = False
    proba_quality = "native"

    def build_estimator(self):
        p = {"n_estimators": 400, "learning_rate": 0.1, "max_depth": 6, "subsample": 0.8,
             "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0,
             "tree_method": "hist", "n_jobs": -1}
        p.update(self.params)
        p.setdefault("random_state", self.seed)  # a params file may carry its own seed
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **p)
