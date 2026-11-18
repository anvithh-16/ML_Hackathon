"""Trip decoder tests on the mini trip. Owner: Member 3."""
from src.candidates.generate import generate
from src.hmm.decode import classifier_only, decode_trip


def test_decode_mini(mini_trip, cfg):
    c = generate(mini_trip, cfg)
    r = decode_trip(mini_trip, c, None, cfg=cfg)
    p = r.preds.set_index("obs_idx").pred_arc_id
    assert list(p[:5]) == [0] * 5 and list(p[5:10]) == [10] * 5 and list(p[10:13]) == [6] * 3
    assert r.preds.segment_id.max() == 1          # 40 s gap -> 2 segments
    assert r.path_arcs[0] == [0, 10]
    assert len(r.preds) == 14


def test_stride_predicts_every_obs(mini_trip, cfg):
    c = generate(mini_trip, cfg)
    h = {"emission_eps": 1e-6, "sigma_m": 5.0, "beta_m": 5.0, "uturn_penalty": 2.0,
         "dijkstra_limit_factor": 3.0, "dijkstra_limit_extra_m": 200, "decode_stride": 3, "max_candidates": 30}
    r = decode_trip(mini_trip, c, None, h, cfg)
    assert (r.preds.pred_arc_id >= 0).all()


def test_classifier_only(mini_trip, cfg):
    c = generate(mini_trip, cfg)
    co = classifier_only(c, None)
    assert co.obs_idx.is_unique and len(co) == 14
