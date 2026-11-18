"""API request/response models (shared with the UI via ui/sample_response.json). Owner: Member 4.

Ground truth NEVER appears here: no reference-route geometry, no labels - only aggregate
metrics computed server-side.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class TripInfo(BaseModel):
    trip_id: str
    n_obs: int


class TripsResponse(BaseModel):
    trips: list[TripInfo]


class Health(BaseModel):
    status: str
    mode: str = Field(description="'model' (trained artifact loaded) or 'distance-only' (no artifact found)")
    model_name: str
    feature_version: str
    eval_root: str


class Metrics(BaseModel):
    top1_accuracy_hmm: float | None = None
    top1_accuracy_classifier: float | None = None
    route_precision: float | None = None
    route_recall: float | None = None
    route_continuity_classifier: float | None = None
    route_continuity_hmm: float | None = None
    n_route_breaks: int = 0
    candidate_miss_rate: float | None = None
    n_labelled_obs: int = 0
    reference_available: bool = False


class PredictResponse(BaseModel):
    trip_id: str
    model_name: str
    model_variant: str
    feature_version: str
    n_obs: int
    n_segments: int
    gps_latlon: list[list[float]]                    # [[lat, lon], ...] (downsampled)
    predicted_route_latlon: list[list[list[float]]]  # one polyline [[lat, lon], ...] per segment
    metrics: Metrics
    timings_s: dict[str, float]
