"""Random Forest candidate classifier. Owner: Member 3.

Bagging of decorrelated trees: lower variance than a single tree, averaged (smoother)
probabilities. No scaling needed. Watch artifact size (deep trees) - compress=3 is applied.
"""
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseCandidateModel


class RFModel(BaseCandidateModel):
    name = "rforest"
    family = "ensemble-bagging"
    needs_scaling = False
    proba_quality = "native"

    def build_estimator(self):
        p = {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 10, "max_features": "sqrt", "n_jobs": -1}
        p.update(self.params)
        return RandomForestClassifier(random_state=self.seed, **p)
