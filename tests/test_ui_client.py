"""UI client + map on the MOCK sample response. Owner: Member 2."""
import json
import sys
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[1] / "ui"
sys.path.insert(0, str(UI))


def test_sample_parses_and_maps():
    from map_view import prediction_map
    resp = json.loads((UI / "sample_response.json").read_text())
    assert "MOCK" in resp["model_name"] or "SAMPLE" in resp["model_name"]
    html = prediction_map(resp)._repr_html_()
    assert "Predicted route" in html and "GPS observations" in html


def test_client_error(monkeypatch):
    import api_client
    import requests

    class R:
        status_code = 404
        text = "nope"
        def json(self):
            return {"detail": "trip not found"}

    monkeypatch.setattr(requests, "post", lambda *a, **k: R())
    with pytest.raises(api_client.ApiError) as e:
        api_client.predict("http://x", "00000001")
    assert e.value.status == 404
