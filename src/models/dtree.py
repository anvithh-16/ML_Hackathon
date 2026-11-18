"""Decision Tree candidate classifier. Owner: Member 2.

Learns axis-aligned rules on the raw features (no scaling needed). Leaf frequencies give
coarse probabilities; depth / min_samples_leaf control overfitting.
"""
from sklearn.tree import DecisionTreeClassifier

from src.models.base import BaseCandidateModel


class DTreeModel(BaseCandidateModel):
    name = "dtree"
    family = "tree"
    needs_scaling = False
    proba_quality = "coarse"

    def build_estimator(self):
        p = {"max_depth": 12, "min_samples_leaf": 50, "criterion": "gini", "max_features": None}
        p.update(self.params)
        return DecisionTreeClassifier(random_state=self.seed, **p)
