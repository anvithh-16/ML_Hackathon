"""CV runner on MOCK data. Owner: Member 2."""
import numpy as np

from src.eval.cv_runner import cross_validate
from src.models.dtree import DTreeModel
from tests.fixtures.synthetic import synthetic_folds


def test_cross_validate_oof(synthetic_df):
    folds = synthetic_folds(synthetic_df)
    r = cross_validate(DTreeModel, {"max_depth": 4}, synthetic_df, folds)
    assert len(r["per_fold"]) == 5
    assert np.isfinite(r["oof"]).all()            # every row scored exactly once out-of-fold
    assert 0 <= r["top1_mean"] <= 1
