"""PathSnap FastAPI backend (Laptop 1). Owner: Member 4.

uvicorn api.main:app --host 0.0.0.0 --port 8000
Endpoints:  GET /health   GET /trips   POST /predict/{trip_id}
Every request and every pipeline stage is logged to the console and logs/api.log so the
evaluator can see the inference requests arriving on this laptop.
"""
from __future__ import annotations

import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.pipeline import predict_trip
from api.schemas import Health, PredictResponse, TripInfo, TripsResponse
from api.scorer import score
from src.common.config import load_config, load_hmm_config, path_of, project_root, resolve
from src.common.logging_utils import get_logger
from src.common.schema import FEATURE_VERSION
from src.data.loader import list_trips

log = get_logger("api")
STATE: dict = {}
TRIP_RE = re.compile(r"^\d{8}$")


def _load_state() -> None:
    cfg = load_config()
    final = project_root() / cfg["api"]["final_artifacts"]
    eval_root = resolve(os.environ.get("PATHSNAP_EVAL_ROOT", cfg["paths"]["eval_root"]))
    hmm_cfg = load_hmm_config(final / "hmm.yaml") if (final / "hmm.yaml").exists() else load_hmm_config()
    model, name, variant = None, "distance-only HMM (no trained model loaded)", "none"
    if (final / "model.joblib").exists():
        from src.models.base import BaseCandidateModel
        import json
        model = BaseCandidateModel.load(final)
        man = json.loads((final / "manifest.json").read_text()) if (final / "manifest.json").exists() else {}
        name, variant = model.name, man.get("variant", "unknown")
    STATE.update(cfg=cfg, hmm=hmm_cfg, model=model, model_name=name, variant=variant, eval_root=eval_root)
    log.info("API ready: model=%s variant=%s eval_root=%s", name, variant, eval_root)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_state()
    yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    client = request.client.host if request.client else "?"
    log.info("REQUEST  %s %s from %s", request.method, request.url.path, client)
    response = await call_next(request)
    log.info("RESPONSE %s %s -> %d in %.2fs", request.method, request.url.path, response.status_code,
             time.perf_counter() - t0)
    return response


@app.get("/health", response_model=Health)
def health():
    return Health(status="ok", mode="model" if STATE["model"] is not None else "distance-only",
                  model_name=STATE["model_name"], feature_version=FEATURE_VERSION, eval_root=str(STATE["eval_root"]))


@app.get("/trips", response_model=TripsResponse)
def trips():
    root: Path = STATE["eval_root"]
    if not root.exists():
        raise HTTPException(404, f"evaluation folder {root} not found on the API laptop")
    out = []
    for tid in list_trips(root, None):
        f = root / tid / f"{tid}.track"
        n = sum(1 for _ in open(f, "rb")) if f.exists() else 0
        out.append(TripInfo(trip_id=tid, n_obs=n))
    return TripsResponse(trips=out)


@app.post("/predict/{trip_id}", response_model=PredictResponse)
def predict(trip_id: str):
    if not TRIP_RE.match(trip_id):
        raise HTTPException(422, "trip_id must be 8 digits")
    path = STATE["eval_root"] / trip_id
    if not path.is_dir():
        raise HTTPException(404, f"trip {trip_id} not found in {STATE['eval_root']}")
    t0 = time.perf_counter()
    try:
        out = predict_trip(path, STATE["model"], STATE["cfg"], STATE["hmm"],
                           STATE["cfg"]["api"]["max_gps_points_response"])
        t1 = time.perf_counter()
        metrics = score(path, out, STATE["cfg"])
        out["timings"]["scoring_s"] = time.perf_counter() - t1
    except Exception as e:  # report, never hide
        log.exception("pipeline failed for %s", trip_id)
        raise HTTPException(500, f"pipeline failed for {trip_id}: {type(e).__name__}: {e}")
    out["timings"]["total_s"] = time.perf_counter() - t0
    res = out["hmm"]
    log.info("trip %s: %d obs, %d segments, top1_hmm=%s top1_clf=%s recall=%s | timings %s", trip_id,
             len(out["trip"].track), int(res.preds.segment_id.max()) + 1, metrics.top1_accuracy_hmm,
             metrics.top1_accuracy_classifier, metrics.route_recall,
             {k: round(v, 2) for k, v in out["timings"].items()})
    return PredictResponse(trip_id=trip_id, model_name=STATE["model_name"], model_variant=STATE["variant"],
                           feature_version=FEATURE_VERSION, n_obs=len(out["trip"].track),
                           n_segments=int(res.preds.segment_id.max()) + 1, gps_latlon=out["gps_latlon"],
                           predicted_route_latlon=out["route_latlon"], metrics=metrics,
                           timings_s={k: float(v) for k, v in out["timings"].items()})
