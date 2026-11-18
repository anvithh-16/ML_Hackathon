"""HTTP client for the PathSnap API. Owner: Member 2. No imports from src/ (Laptop 2 has only ui/)."""
from __future__ import annotations

import requests


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


def _get(url: str, timeout: float) -> dict:
    try:
        r = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        raise ApiError(0, f"cannot reach {url}: {e}") from e
    if r.status_code != 200:
        raise ApiError(r.status_code, _detail(r))
    return r.json()


def _detail(r) -> str:
    try:
        return str(r.json().get("detail", r.text))
    except ValueError:
        return r.text[:300]


def get_health(base_url: str, timeout: float = 5) -> dict:
    return _get(base_url.rstrip("/") + "/health", timeout)


def get_trips(base_url: str, timeout: float = 15) -> list[dict]:
    return _get(base_url.rstrip("/") + "/trips", timeout)["trips"]


def predict(base_url: str, trip_id: str, timeout: float = 300) -> dict:
    url = f"{base_url.rstrip('/')}/predict/{trip_id}"
    try:
        r = requests.post(url, timeout=timeout)
    except requests.RequestException as e:
        raise ApiError(0, f"cannot reach {url}: {e}") from e
    if r.status_code != 200:
        raise ApiError(r.status_code, _detail(r))
    return r.json()
