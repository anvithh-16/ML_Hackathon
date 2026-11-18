"""Candidate generation tests. Owner: Member 1."""
from src.candidates.generate import generate
from src.common.schema import CAND_COLUMNS


def test_candidates(mini_trip, cfg):
    c = generate(mini_trip, cfg)
    assert list(c.columns) == CAND_COLUMNS
    first = c[c.obs_idx == 0]
    assert {0, 1} <= set(first.arc_id)                     # true arc and its reverse twin
    assert c.groupby("obs_idx").size().max() <= cfg["candidates"]["max_per_obs"]
    assert (c.dist_m >= 0).all() and c.proj_t.between(0, 1).all()
    off = c[c.obs_idx == 13]                               # 80 m off-road -> fallback candidates only
    assert len(off) >= 1 and (off.dist_m > cfg["candidates"]["radius_m"]).all()
    assert (c.groupby("obs_idx").cand_rank.min() == 0).all()
