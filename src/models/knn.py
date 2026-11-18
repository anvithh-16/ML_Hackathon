"""K-Nearest Neighbours candidate classifier. Owner: Member 2.

Instance-based: a candidate is scored by the labels of similar training candidates.
Distances need comparable feature scales -> StandardScaler inside the pipeline.
Probabilities are vote fractions (coarse); prediction is the slowest of the five models,
so the base class predicts in chunks of 100k rows.
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseCandidateModel


class KNNModel(BaseCandidateModel):
    name = "knn"
    family = "instance-based"
    needs_scaling = True
    proba_quality = "coarse"

    def build_estimator(self):
        p = {"n_neighbors": 25, "weights": "distance", "p": 2, "algorithm": "auto", "n_jobs": -1}
        p.update(self.params)
        return make_pipeline(StandardScaler(), KNeighborsClassifier(**p))
