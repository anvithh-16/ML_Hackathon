"""Every registered model through fit / proba / save / load on MOCK data. Owner: Member 4."""
import numpy as np
import pytest

from src.common.schema import FEATURE_COLUMNS, SchemaError
from src.models.registry import MODEL_REGISTRY, available_models

FAST = {"rforest": {"n_estimators": 20}, "xgb": {"n_estimators": 30}, "knn": {"n_neighbors": 5}}


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_contract(name, synthetic_df, tmp_path):
    if name not in available_models():
        pytest.skip(f"{name} module not merged yet")
    from src.models.registry import get_model_class
    cls = get_model_class(name)
    X, y = synthetic_df[FEATURE_COLUMNS], synthetic_df["y"].to_numpy()
    m = cls(FAST.get(name, {}), seed=42).fit(X, y, groups=synthetic_df.trip_id.to_numpy())
    p = m.predict_proba(X)
    assert p.shape == (len(X),)
    assert np.all((p >= 1e-6) & (p <= 1 - 1e-6))
    assert m.train_time_s is not None and m.train_time_s >= 0
    m.save(tmp_path / name)
    m2 = cls.load(tmp_path / name)
    np.testing.assert_allclose(m2.predict_proba(X), p)
    with pytest.raises(SchemaError):
        m.predict_proba(X[list(reversed(FEATURE_COLUMNS))])
    s = m.score_candidates(synthetic_df[["trip_id", "obs_idx", "arc_id"]], X)
    assert list(s.columns) == ["trip_id", "obs_idx", "arc_id", "score"]
