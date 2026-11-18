# 7b. Evaluator UI  (owner: Member 2)

Code: `ui/app.py`, `ui/api_client.py`, `ui/map_view.py` · Dependencies for the UI laptop only: `ui/requirements-ui.txt`
(streamlit, folium, requests) · Tests: `tests/test_ui_client.py`

## What the UI does

The UI is a Streamlit page that runs on the second laptop and talks to the API on the first laptop over HTTP. It contains no
model, no data and no code from `src/`; it only shows what the API returns. The evaluator's flow is:

1. The sidebar holds the API address (default `http://localhost:8000`, or the environment variable `PATHSNAP_API_URL`). The
   page calls `/health` and shows either a green "API online · model: xgb · mode: model" box or a red error.
2. The page calls `/trips` and fills a drop-down with the evaluation trips found on the API laptop, each with its number of GPS
   points.
3. The evaluator chooses a trip and presses **Run map matching**. The page sends `POST /predict/<trip_id>`.
4. When the answer arrives the page shows eight metric cards, a map and a collapsible "Raw API response". The result is kept in
   `st.session_state`, so changing a widget does not send the request again.

## Screenshots

Taken on one machine with the real stack (API in model mode with the exported XGBoost, UI in a browser), using a
student_test trip as a stand-in for an evaluator trip. The two-laptop rehearsal over Wi-Fi was not done for these images.

| Screenshot | What it shows |
|---|---|
| `docs/report/screenshots/ui_1_trip_list.png` | the connection sidebar and the trip drop-down (10 trips) |
| `docs/report/screenshots/ui_2_result_00000007.png` | metric cards and the map for trip 00000007 |
| `docs/report/screenshots/ui_3_api_unreachable.png` | the message shown when the API address is wrong |

The numbers in the result screenshot are the API's response for that trip, saved in `results/ui/ui_demo_00000007.json`
(polylines left out):

| Card | Value |
|---|---|
| Top-1 accuracy (HMM) | 99.34% |
| Top-1 accuracy (classifier only) | 95.35% |
| Route precision / recall | 98.37% / 98.37% |
| Route continuity (HMM / classifier only) | 100.00% / 99.63% |
| Segments / breaks | 1 / 0 |
| Round-trip time | 0.2 s (server-side total 0.179 s: HMM 0.096 s, scoring 0.035 s, candidates 0.030 s) |

These are results for one trip only and are not the overall performance. The overall student_test numbers are in
section 5 and `results/hmm/clf_vs_hmm.csv`.

The map draws every GPS observation as a small blue circle and the predicted route as a red line, one line per route segment
(a new segment starts at a long time gap, a large jump, or a point with no candidate road). The map has a layer control to
switch the two layers on and off and zooms to fit the route. The base map is OpenStreetMap, so the UI laptop needs internet
access for the tiles.

## What the UI receives, and what it does not

The API answer (`api/schemas.py`) contains: trip id, model name and variant, feature version, number of observations and
segments, the GPS points as `[lat, lon]`, the predicted route as `[lat, lon]` lines, a `metrics` block and server-side stage
timings. It contains **no reference route and no labels**. The accuracy metrics are computed on the API laptop, which is the
only place that reads the `.route` file (`api/scorer.py`; `api/pipeline.py` never opens it, and a test checks this). The UI
therefore cannot leak or alter the ground truth, and the model never sees it. If a trip has no reference route on the server,
the response marks `reference_available` as false and the UI shows an information box instead of accuracy numbers.

## Errors and timeouts

`ui/api_client.py` turns every failure into one exception type, `ApiError(status, message)`:
- connection refused, wrong address or timeout: status 0 with the underlying reason (screenshot 3);
- unknown trip: HTTP 404 with the server's message (checked in `tests/test_ui_client.py` and against the live API);
- bad trip id: HTTP 422; server error: HTTP 500 with the message.

The page shows these as red error boxes and does not crash. Timeouts are 5 s for `/health`, 15 s for `/trips` and 300 s for
`/predict`, because a long trip can take a few seconds on the server (0.1 to 1.7 s per trip for the 10 student_test trips in
our run). The first request after the API starts is slower because the modules are imported and the road graph is built, so we
send one warm-up request before the evaluation.

## Mock mode and tests

`streamlit run ui/app.py -- --mock` reads `ui/sample_response.json`, which was built from the small synthetic trip used in the
tests. It shows a "MOCK" banner and is only for developing the UI before the API existed; its numbers are never reported.
`tests/test_ui_client.py` checks that the sample response becomes a map with both layers and that the client raises
`ApiError(404)` for an unknown trip.
