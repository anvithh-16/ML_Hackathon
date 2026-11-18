"""PathSnap evaluator UI (Laptop 2). Owner: Member 2.

PATHSNAP_API_URL=http://<LAPTOP1_IP>:8000 streamlit run ui/app.py
streamlit run ui/app.py -- --mock        # uses ui/sample_response.json (MOCK, not a result)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_client import ApiError, get_health, get_trips, predict  # noqa: E402
from map_view import prediction_map  # noqa: E402

MOCK = "--mock" in sys.argv
SAMPLE = Path(__file__).resolve().parent / "sample_response.json"

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap — GPS map matching")

with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("API URL (Laptop 1)", os.environ.get("PATHSNAP_API_URL", "http://localhost:8000"))
    if MOCK:
        st.warning("MOCK MODE: showing ui/sample_response.json — not a real prediction.")
        health = {"status": "mock", "mode": "mock", "model_name": "SAMPLE"}
    else:
        try:
            health = get_health(api_url)
            st.success(f"API online · model: {health['model_name']} · mode: {health['mode']}")
        except ApiError as e:
            health = None
            st.error(f"API not reachable: {e}")

if MOCK:
    sample = json.loads(SAMPLE.read_text())
    trips = [{"trip_id": sample["trip_id"], "n_obs": sample["n_obs"]}]
elif health:
    try:
        trips = get_trips(api_url)
    except ApiError as e:
        st.error(str(e))
        trips = []
else:
    trips = []

if not trips:
    st.info("No trips available yet. Check the API URL and that the evaluation folder exists on Laptop 1.")
    st.stop()

labels = {f"{t['trip_id']}  ({t['n_obs']} GPS points)": t["trip_id"] for t in trips}
choice = st.selectbox(f"Select one of the {len(trips)} evaluation trips", list(labels))
trip_id = labels[choice]

if st.button("Run map matching", type="primary"):
    with st.spinner(f"Sending trip {trip_id} to the API ..."):
        t0 = time.perf_counter()
        try:
            resp = json.loads(SAMPLE.read_text()) if MOCK else predict(api_url, trip_id)
            st.session_state[trip_id] = (resp, time.perf_counter() - t0)
        except ApiError as e:
            st.error(str(e))

if trip_id in st.session_state:
    resp, elapsed = st.session_state[trip_id]
    m = resp["metrics"]
    fmt = lambda x: "n/a" if x is None else f"{100 * x:.2f}%"
    c = st.columns(4)
    c[0].metric("Top-1 accuracy (HMM)", fmt(m.get("top1_accuracy_hmm")))
    c[1].metric("Top-1 accuracy (classifier only)", fmt(m.get("top1_accuracy_classifier")))
    c[2].metric("Route precision", fmt(m.get("route_precision")))
    c[3].metric("Route recall", fmt(m.get("route_recall")))
    c = st.columns(4)
    c[0].metric("Route continuity (HMM)", fmt(m.get("route_continuity_hmm")))
    c[1].metric("Route continuity (classifier)", fmt(m.get("route_continuity_classifier")))
    c[2].metric("Segments / breaks", f"{resp['n_segments']} / {m.get('n_route_breaks', 0)}")
    c[3].metric("Round-trip time", f"{elapsed:.1f} s")
    if not m.get("reference_available"):
        st.info("No reference route on the server for this trip: accuracy metrics are not available.")
    st.caption(f"Model: {resp['model_name']} ({resp['model_variant']}) · features {resp['feature_version']} · "
               f"{resp['n_obs']} GPS points · metrics computed on the API laptop")
    components.html(prediction_map(resp)._repr_html_(), height=620)
    with st.expander("Raw API response"):
        st.json({k: v for k, v in resp.items() if k not in ("gps_latlon", "predicted_route_latlon")})
