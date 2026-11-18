"""Shared pytest fixtures. Owner: Member 4."""
from pathlib import Path

import pytest

from src.common.config import load_config, path_of

MINI_ROOT = Path(__file__).resolve().parent / "fixtures" / "mini_trip"


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def mini_dir():
    return MINI_ROOT / "00000999"


@pytest.fixture(scope="session")
def mini_trip(mini_dir):
    from src.data.geo import project
    from src.data.loader import load_trip
    return project(load_trip(mini_dir, "mock"))


@pytest.fixture(scope="session")
def data_root(cfg):
    root = path_of(cfg, "data_root")
    if not (root / "train").exists():
        pytest.skip(f"real dataset not found at {root}")
    return root


@pytest.fixture(scope="session")
def synthetic_df():
    from tests.fixtures.synthetic import make_synthetic
    return make_synthetic(n_trips=10, obs_per_trip=30, seed=7)
