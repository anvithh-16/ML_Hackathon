"""Organiser data-use rules. Owner: Member 2."""
import inspect
import json

import pytest

from src.common.config import project_root
from src.eval import cv_runner


def _folds():
    p = project_root() / "configs" / "folds.json"
    if not p.exists():
        pytest.skip("configs/folds.json not generated yet (Member 1: python -m src.data.make_folds)")
    return json.loads(p.read_text())


def test_folds_cover_80_train_trips_only(data_root):
    f = _folds()
    from src.data.loader import list_trips
    assert set(f["folds"]) == set(list_trips(data_root, "train"))
    assert not set(f["folds"]) & set(list_trips(data_root, "student_test"))
    assert sorted(set(f["folds"].values())) == [0, 1, 2, 3, 4]


def test_runner_never_touches_eval_root():
    assert "eval_root" not in inspect.getsource(cv_runner)


def test_cross_validate_asserts_disjoint_trips():
    assert "trip leakage" in inspect.getsource(cv_runner.cross_validate)
