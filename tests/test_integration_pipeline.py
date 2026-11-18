"""End-to-end: features -> tiny model -> API pipeline on the MOCK mini trip. Owner: Member 4."""
from api.pipeline import predict_trip
from src.candidates.generate import generate
from src.common.config import load_hmm_config
from src.common.schema import FEATURE_COLUMNS
from src.data.loader import load_route
from src.features.build_features import build
from src.labels.align_route import align
from src.models.logreg import LogRegModel


def test_end_to_end(mini_trip, mini_dir, cfg, tmp_path):
    f = build(mini_trip, generate(mini_trip, cfg), align(mini_trip, load_route(mini_dir), cfg), cfg)
    f = f[f.labelled]
    m = LogRegModel().fit(f[FEATURE_COLUMNS], f.y.to_numpy())
    m.save(tmp_path / "m")
    m = LogRegModel.load(tmp_path / "m")
    out = predict_trip(mini_dir, m, cfg, load_hmm_config())
    assert len(out["hmm"].preds) == 14
    assert out["route_latlon"] and out["gps_latlon"]
