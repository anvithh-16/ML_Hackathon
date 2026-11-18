"""API tests against the MOCK mini trip. Owner: Member 4."""
import inspect

from fastapi.testclient import TestClient

import api.pipeline


def test_endpoints(monkeypatch, mini_dir):
    monkeypatch.setenv("PATHSNAP_EVAL_ROOT", str(mini_dir.parent))
    from api.main import app
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        trips = c.get("/trips").json()["trips"]
        assert [t["trip_id"] for t in trips] == ["00000999"]
        r = c.post("/predict/00000999")
        assert r.status_code == 200
        j = r.json()
        assert j["n_obs"] == 14 and j["metrics"]["reference_available"]
        assert "label" not in str(list(j.keys()))
        assert c.post("/predict/12345678").status_code == 404
        assert c.post("/predict/abc").status_code == 422


def test_pipeline_never_reads_route():
    assert "load_route" not in inspect.getsource(api.pipeline)
