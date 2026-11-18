"""PathSnap shared contract (FROZEN after kickoff).

Owner: Member 4, co-approved by Member 1. Any change needs a PR approved by
both and a bump of FEATURE_VERSION (old model artifacts then refuse to load).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

FEATURE_VERSION = "v1"

# ---------------------------------------------------------------- raw / trip tables
TRACK_COLUMNS = ["obs_idx", "lon", "lat", "t", "x", "y"]
NODE_COLUMNS = ["node_id", "lon", "lat", "x", "y"]
ARC_COLUMNS = ["arc_id", "u", "v", "length_m", "bearing_deg", "reverse_arc_id"]

# ---------------------------------------------------------------- pipeline tables
CAND_COLUMNS = ["trip_id", "obs_idx", "arc_id", "dist_m", "proj_t", "cand_rank"]
LABEL_COLUMNS = ["trip_id", "obs_idx", "label_arc_id", "route_pos", "align_dist_m"]
KEY_COLUMNS = ["trip_id", "obs_idx", "arc_id"]

FEATURE_COLUMNS = [
    "dist_m",
    "dist_rank",
    "dist_minus_min",
    "heading_diff_deg",
    "heading_valid",
    "speed_mps",
    "dt_prev_s",
    "arc_length_m",
    "proj_t",
    "n_candidates",
    "u_degree",
    "v_degree",
    "prev_obs_dist_same_arc",
    "next_obs_dist_same_arc",
    "reverse_exists",
]
TARGET_COLUMN = "y"
LABELLED_COLUMN = "labelled"
FEATURE_TABLE_EXTRA = [TARGET_COLUMN, LABELLED_COLUMN, "label_arc_id"]

SCORE_COLUMNS = ["trip_id", "obs_idx", "arc_id", "score"]
PRED_COLUMNS = ["trip_id", "obs_idx", "pred_arc_id", "segment_id"]

PROBA_CLIP = 1e-6
TRIP_ID_LEN = 8


class SchemaError(ValueError):
    """Raised when a table does not match the shared contract."""


@dataclass
class Trip:
    trip_id: str
    partition: str | None
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    epsg: int | None = None  # None until src.data.geo.project() has run
    meta: dict = field(default_factory=dict)

    @property
    def projected(self) -> bool:
        return self.epsg is not None


def check_columns(df: pd.DataFrame, expected: list[str], name: str = "table") -> None:
    """Raise SchemaError if df lacks columns or has them in a different order."""
    cols = list(df.columns)
    missing = [c for c in expected if c not in cols]
    if missing:
        raise SchemaError(f"{name}: missing columns {missing}")
    sub = [c for c in cols if c in expected]
    if sub != expected:
        raise SchemaError(f"{name}: column order differs from contract.\n expected {expected}\n got      {sub}")


def check_trip_id(trip_id: str) -> str:
    trip_id = str(trip_id)
    if len(trip_id) != TRIP_ID_LEN or not trip_id.isdigit():
        raise SchemaError(f"trip_id must be an {TRIP_ID_LEN}-digit string, got {trip_id!r}")
    return trip_id
