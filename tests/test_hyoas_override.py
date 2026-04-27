import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
from qqq_cycle.data_contracts.hyoas_override import HyoasOverrideStore

@pytest.fixture
def valid_csv(tmp_path):
    path = tmp_path / "valid.csv"
    content = (
        "trade_date,value,source_name,source_timestamp,license_tag,notes\n"
        "2000-01-03,4.99,ICE_BofA,2026-04-27T10:00:00Z,internal,initial\n"
        "2000-01-04,5.03,ICE_BofA,2026-04-27T10:00:00Z,internal,\n"
        "2000-01-05,5.05,ICE_BofA,2026-04-27T10:00:00Z,internal,\n"
    )
    path.write_text(content)
    return path

@pytest.fixture
def long_history_csv(tmp_path):
    path = tmp_path / "long.csv"
    dates = pd.date_range("2000-01-01", periods=600, freq="D")
    df = pd.DataFrame({
        "trade_date": dates,
        "value": np.random.randn(600),
        "source_name": "ICE_BofA",
        "source_timestamp": "2026-04-27T10:00:00Z",
        "license_tag": "internal"
    })
    df.to_csv(path, index=False)
    return path

def test_hyoas_override_valid(long_history_csv):
    store = HyoasOverrideStore(long_history_csv)
    series = store.to_series()
    assert len(series) == 600
    assert series.name == "BAMLH0A0HYM2"
    
    manifest = store.generate_manifest()
    assert manifest["coverage_ok"] is True
    assert manifest["row_count"] == 600
    assert manifest["source_name"] == "ICE_BofA"

def test_hyoas_override_insufficient_coverage(valid_csv):
    store = HyoasOverrideStore(valid_csv)
    manifest = store.generate_manifest()
    assert manifest["coverage_ok"] is False
    assert "insufficient coverage" in manifest["reason_if_rejected"].lower()

def test_hyoas_override_missing_column(tmp_path):
    path = tmp_path / "missing_col.csv"
    path.write_text("trade_date,value,source_name\n2000-01-01,1.0,test")
    with pytest.raises(ValueError, match="missing required columns"):
        HyoasOverrideStore(path)

def test_hyoas_override_duplicate_dates(tmp_path):
    path = tmp_path / "dup.csv"
    path.write_text(
        "trade_date,value,source_name,source_timestamp,license_tag\n"
        "2000-01-01,1.0,test,2026,tag\n"
        "2000-01-01,2.0,test,2026,tag"
    )
    with pytest.raises(ValueError, match="duplicate trade_date"):
        HyoasOverrideStore(path)

def test_hyoas_override_non_numeric_value(tmp_path):
    path = tmp_path / "nan.csv"
    path.write_text(
        "trade_date,value,source_name,source_timestamp,license_tag\n"
        "2000-01-01,BAD,test,2026,tag"
    )
    with pytest.raises(ValueError, match="non-numeric or non-finite value"):
        HyoasOverrideStore(path)

def test_hyoas_override_non_increasing_dates(tmp_path):
    path = tmp_path / "unsorted.csv"
    path.write_text(
        "trade_date,value,source_name,source_timestamp,license_tag\n"
        "2000-01-02,1.0,test,2026,tag\n"
        "2000-01-01,2.0,test,2026,tag"
    )
    with pytest.raises(ValueError, match="trade_date must be strictly increasing"):
        HyoasOverrideStore(path)
