"""Feature tests incl. route-independence. Owner: Member 1."""
import numpy as np

from src.candidates.generate import generate
from src.common.schema import FEATURE_COLUMNS, KEY_COLUMNS
from src.data.loader import load_route
from src.features.build_features import build
from src.labels.align_route import align


def test_features(mini_trip, mini_dir, cfg):
    c = generate(mini_trip, cfg)
    f0 = build(mini_trip, c, None, cfg)
    assert list(f0.columns) == KEY_COLUMNS + FEATURE_COLUMNS
    assert np.isfinite(f0[FEATURE_COLUMNS].to_numpy()).all()
    lab = align(mini_trip, load_route(mini_dir), cfg)
    f1 = build(mini_trip, c, lab, cfg)
    # labels never change features (route-independence)
    np.testing.assert_array_equal(f0[FEATURE_COLUMNS].to_numpy(), f1[FEATURE_COLUMNS].to_numpy())
    assert f1.groupby("obs_idx").y.sum().max() == 1
    assert f1[f1.obs_idx == 13].labelled.eq(False).all()
    # the correct direction has the smaller heading difference
    o = f1[f1.obs_idx == 2].set_index("arc_id")
    assert o.loc[0, "heading_diff_deg"] < o.loc[1, "heading_diff_deg"]


def test_no_route_reads_in_feature_code():
    import inspect
    import src.candidates.generate as g
    import src.features.build_features as b
    assert "load_route" not in inspect.getsource(g)
    src = inspect.getsource(b.build)
    assert "load_route" not in src and ".route" not in src
