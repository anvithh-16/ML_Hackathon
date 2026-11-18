"""Contract sanity. Owner: Member 4."""
import pandas as pd
import pytest

from src.common import schema as S


def test_feature_contract():
    assert S.FEATURE_VERSION == "v1"
    assert len(S.FEATURE_COLUMNS) == 15
    assert len(set(S.FEATURE_COLUMNS)) == 15


def test_check_columns_order():
    df = pd.DataFrame(columns=list(reversed(S.CAND_COLUMNS)))
    with pytest.raises(S.SchemaError):
        S.check_columns(df, S.CAND_COLUMNS)


def test_trip_id():
    assert S.check_trip_id("00000001") == "00000001"
    with pytest.raises(S.SchemaError):
        S.check_trip_id("1")
