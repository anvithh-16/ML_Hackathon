"""Logistic Regression candidate classifier. Owner: Member 1.

Linear baseline with well-behaved probabilities. Needs feature scaling -> StandardScaler inside
the pipeline (the shared feature table stays unscaled for every model).
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseCandidateModel


class LogRegModel(BaseCandidateModel):
    name = "logreg"
    family = "linear"
    needs_scaling = True
    proba_quality = "native"

    def build_estimator(self):
        p = {"C": 1.0, "max_iter": 2000, "solver": "lbfgs", "class_weight": None}
        p.update(self.params)
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=self.seed, **p))
