"""T8: CsvWeightStore as-of semantics and weight accuracy tests."""

from __future__ import annotations

import io
import textwrap

import pandas as pd
import pytest

from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError
from qqq_cycle.data_contracts.weights import CsvWeightStore


def _store_from_csv(text: str) -> CsvWeightStore:
    """Build a CsvWeightStore from a CSV string (no temp files needed)."""
    df = pd.read_csv(io.StringIO(textwrap.dedent(text).strip()))
    store = CsvWeightStore.__new__(CsvWeightStore)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    df["asof_timestamp"] = pd.to_datetime(df["asof_timestamp"], utc=False)
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df["weight"] = df["weight"].astype(float)
    store._df = df
    return store


# ── T8.1: asof weight lookup returns correct value ────────────────────────────

def test_asof_weight_lookup_returns_correct_value():
    """get_weights returns the correct ticker→weight mapping for a given date."""
    store = _store_from_csv("""
        trade_date,ticker,weight,asof_timestamp
        2021-03-19,AAPL,0.115,2021-03-19T16:00:00
        2021-03-19,MSFT,0.097,2021-03-19T16:00:00
        2021-03-19,NVDA,0.050,2021-03-19T16:00:00
    """)
    weights = store.get_weights(
        pd.Timestamp("2021-03-19"), asof=pd.Timestamp("2021-03-19T23:59:59")
    )
    assert weights == pytest.approx({"AAPL": 0.115, "MSFT": 0.097, "NVDA": 0.050})


# ── T8.2: future weight not returned for prior dates ─────────────────────────

def test_future_weight_not_returned():
    """A weight published Monday is not visible with asof=Friday 16:00."""
    store = _store_from_csv("""
        trade_date,ticker,weight,asof_timestamp
        2021-03-19,AAPL,0.115,2021-03-19T16:00:00
        2021-03-19,MSFT,0.097,2021-03-19T16:00:00
        2021-03-19,TSLA,0.030,2021-03-22T09:00:00
    """)
    # Friday EOD: TSLA row (published Monday morning) must NOT appear.
    weights = store.get_weights(
        pd.Timestamp("2021-03-19"), asof=pd.Timestamp("2021-03-19T23:59:59")
    )
    assert "TSLA" not in weights, "future weight leaked through strict asof filter"
    assert set(weights.keys()) == {"AAPL", "MSFT"}

    # After Monday publish: TSLA is visible.
    weights_mon = store.get_weights(
        pd.Timestamp("2021-03-19"), asof=pd.Timestamp("2021-03-22T23:59:59")
    )
    assert "TSLA" in weights_mon


# ── T8.3: missing trade date raises DataNotAvailableError ─────────────────────

def test_missing_trade_date_raises():
    """DataNotAvailableError raised when no weight rows match the requested date."""
    store = _store_from_csv("""
        trade_date,ticker,weight,asof_timestamp
        2021-03-19,AAPL,0.115,2021-03-19T16:00:00
    """)
    with pytest.raises(DataNotAvailableError):
        store.get_weights(
            pd.Timestamp("2021-03-26"), asof=pd.Timestamp("2021-03-26T23:59:59")
        )


# ── T8.4: strict asof filter on weights ──────────────────────────────────────

def test_strict_asof_filter_on_weights():
    """Rows whose asof_timestamp > asof are invisible even on the correct trade_date."""
    store = _store_from_csv("""
        trade_date,ticker,weight,asof_timestamp
        2021-03-19,AAPL,0.115,2021-03-19T16:00:00
        2021-03-19,MSFT,0.097,2021-03-20T09:00:00
    """)
    weights = store.get_weights(
        pd.Timestamp("2021-03-19"), asof=pd.Timestamp("2021-03-19T16:00:00")
    )
    assert "AAPL" in weights
    assert "MSFT" not in weights

    weights2 = store.get_weights(
        pd.Timestamp("2021-03-19"), asof=pd.Timestamp("2021-03-20T23:59:59")
    )
    assert "MSFT" in weights2


# ── T8.5: weights sum approximately to 1 (smoke test on seed data) ────────────

def test_seed_weights_reasonable():
    """Seed weights should be positive and sum close to 1.0 for any given day."""
    import pandas as pd
    from pathlib import Path
    seed_path = Path("cache/micro/weights.csv")
    if not seed_path.exists():
        pytest.skip("seed data not present; run scripts/seed_micro_data.py")
    store_real = CsvWeightStore(seed_path)
    day = pd.Timestamp("2023-06-02")
    asof = day + pd.Timedelta(hours=23, minutes=59, seconds=59)
    weights = store_real.get_weights(day, asof=asof)
    assert len(weights) > 0
    assert all(w > 0 for w in weights.values())
    total = sum(weights.values())
    assert 0.5 < total < 1.5, f"weights sum {total} is implausible"
