<!-- AI-assisted first draft (Claude Code). Member 4: rewrite in your own words before submission. Fill the TODO parts after the two-laptop rehearsal. -->
# 7. API and deployment  (owner: Member 4)

Code: `api/main.py`, `api/pipeline.py`, `api/scorer.py`, `api/schemas.py` · Runbook: `docs/DEPLOYMENT.md` ·
Loaded model: `artifacts/final/` (written only by `python -m src.export.export_artifacts`).

## Architecture

```
Laptop 2 (evaluator)                         Laptop 1 (server)
Streamlit UI  ──HTTP over Wi-Fi/hotspot──▶  FastAPI (uvicorn, 0.0.0.0:8000)
  ui/app.py                                    model (artifacts/final/model.joblib = xgb__default)
  no model, no data                            HMM config (artifacts/final/hmm.yaml)
                                               evaluator trips (data/eval/<trip_id>/)
                                               scorer: the only code that reads .route
```

All computation and all data stay on Laptop 1. The UI only displays what the API returns (section 7b).

## Endpoints

| Endpoint | Returns | Errors |
|---|---|---|
| `GET /health` | status, mode (`model` when `artifacts/final/model.joblib` is loaded, otherwise `distance-only`), model name, feature version, evaluation folder | — |
| `GET /trips` | trip ids found in the evaluation folder with their number of GPS points | 404 if the folder is missing |
| `POST /predict/{trip_id}` | GPS points and predicted route as [lat, lon] polylines, number of route segments, aggregate metrics, per-stage timings | 422 if the id is not 8 digits, 404 if the trip is unknown, 500 with the error message if a stage fails |

## Request flow for `POST /predict/{trip_id}` (`api/pipeline.py`)

1. **Load + project:** read `.track`, `.nodes`, `.arcs` and project to metres in the trip's own UTM zone.
2. **Candidates:** arcs within 50 m of each GPS point (at most 30); if none, the 3 nearest arcs within 200 m.
3. **Features:** the 15 `FEATURE_COLUMNS`, computed with `labels=None` (no ground truth is available or needed).
4. **Model:** XGBoost gives P(correct arc) for every candidate.
5. **HMM / Viterbi:** these probabilities are the emissions; transitions use road-network distance; Viterbi returns one
   connected route (a new segment starts only when consecutive points are more than 30 s or 500 m apart).
6. **Response:** GPS points (downsampled to at most 3 000) and the route polylines converted back to latitude/longitude.
7. **Scoring (`api/scorer.py`):** only after prediction, the server reads the `.route` file if one exists, aligns it to the
   GPS points and computes Top-1 (HMM and classifier-only), route precision/recall, continuity and candidate miss rate.
   Only these aggregate numbers are sent back. If an evaluator trip has no `.route`, the metrics are null.

## Where ground truth lives

The reference route is read only in `api/scorer.py`, and only after the prediction is finished. `api/pipeline.py` never
reads it (`tests/test_api.py` checks that `load_route` does not appear in the pipeline), the response schema
(`api/schemas.py`) has no field for labels or reference geometry, and the UI never receives the reference route. So the
prediction cannot use the answer, and the answer never leaves the server except as aggregate scores.

## Logging

A middleware logs every request and response with the client IP, status code and duration, and `/predict` also logs the
number of observations, the metrics and the per-stage timings. The console output is saved with
`uvicorn ... 2>&1 | tee logs/api_console.log`, so the evaluator can see requests from Laptop 2 arriving on Laptop 1.

## Measured timings

Single-machine run on student_test trip 00000007 (1 354 GPS points), API in model mode with `xgb__default`
(`results/ui/ui_demo_00000007.json`):

| Stage | Time (s) |
|---|---:|
| Load + project | 0.006 |
| Candidates | 0.030 |
| Features | 0.003 |
| Model | 0.004 |
| HMM / Viterbi | 0.096 |
| Scoring | 0.035 |
| **Total** | **0.179** |

The HMM is the slowest stage (shortest-path searches between candidates); the XGBoost model itself takes 4 ms.
Metrics for this trip: Top-1 0.953 classifier-only → 0.993 after the HMM, 1 route segment, no breaks. This recorded run
used the earlier export with `uturn_penalty` 2.0; after re-exporting with the tuned decoder config (5.0, PR #27) the API
gives 0.994 for this trip, identical to the offline value in `results/hmm/per_trip.csv`, and all 10 student_test trips
match the offline HMM Top-1 exactly.

TODO after the rehearsal: add the three demo trips from `results/logs/api_demo.log` (trip id, n_obs, total time, Top-1 HMM).

## Two-laptop deployment

Steps are in `docs/DEPLOYMENT.md`: Laptop 1 runs `uvicorn api.main:app --host 0.0.0.0 --port 8000`, finds its address
with `ipconfig getifaddr en0`, and Laptop 2 starts the UI with `PATHSNAP_API_URL=http://<Laptop-1-IP>:8000`. Binding to
`0.0.0.0` (not `127.0.0.1`) is what makes the API reachable from the other laptop; campus Wi-Fi may block
laptop-to-laptop traffic, in which case both laptops join a phone hotspot.

TODO after the rehearsal: screenshot of the Laptop 1 terminal showing the `REQUEST`/`RESPONSE` lines from Laptop 2's IP
(`docs/report/screenshots/`), and one sentence on any problem met and how it was fixed.
