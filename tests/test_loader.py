"""Loader tests. Owner: Member 1."""
import shutil

import pytest

from src.data.loader import TripFormatError, list_trips, load_route, load_trip


def test_mini_trip(mini_dir):
    t = load_trip(mini_dir)
    assert t.trip_id == "00000999"
    assert (len(t.track), len(t.nodes), len(t.arcs)) == (14, 6, 14)
    assert list(load_route(mini_dir)) == [0, 10, 6]


def test_missing_file(mini_dir, tmp_path):
    d = tmp_path / "00000998"
    shutil.copytree(mini_dir, d)
    for f in d.iterdir():
        f.rename(d / f.name.replace("00000999", "00000998"))
    (d / "00000998.arcs").unlink()
    with pytest.raises(TripFormatError):
        load_trip(d)


def test_list_trips_ignores_junk(tmp_path):
    (tmp_path / "00000001").mkdir()
    (tmp_path / "__MACOSX").mkdir()
    (tmp_path / ".DS_Store").write_text("x")
    assert list_trips(tmp_path) == ["00000001"]


@pytest.mark.realdata
def test_known_trip_counts(data_root):
    assert len(list_trips(data_root, "train")) == 80
    assert len(list_trips(data_root, "student_test")) == 10
    t = load_trip(data_root / "train" / "00000001")
    assert (len(t.track), len(t.nodes), len(t.arcs)) == (1315, 88809, 165790)
    assert len(load_route(data_root / "train" / "00000001")) == 525
