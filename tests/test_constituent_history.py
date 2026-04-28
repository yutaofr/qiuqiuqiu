"""T7: CsvConstituentStore as-of semantics and membership accuracy tests."""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from qqq_cycle.data_contracts.constituents import CsvConstituentStore
from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError


def _store_from_csv(text: str) -> CsvConstituentStore:
    """Build a CsvConstituentStore from a CSV string (no temp files needed)."""
    df = pd.read_csv(io.StringIO(textwrap.dedent(text).strip()))
    store = CsvConstituentStore.__new__(CsvConstituentStore)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    df["asof_timestamp"] = pd.to_datetime(df["asof_timestamp"], utc=False)
    df["ticker"] = df["ticker"].str.strip().str.upper()
    store._df = df
    return store


# ── T7.1: cross-section returns correct membership ───────────────────────────

def test_cross_section_returns_correct_membership():
    """get_snapshot returns exactly the members present on a given trade date."""
    store = _store_from_csv("""
        trade_date,ticker,asof_timestamp
        2021-03-19,AAPL,2021-03-19T16:00:00
        2021-03-19,MSFT,2021-03-19T16:00:00
        2021-03-19,NVDA,2021-03-19T16:00:00
    """)
    snap = store.get_snapshot(
        pd.Timestamp("2021-03-19"), asof=pd.Timestamp("2021-03-19T23:59:59")
    )
    assert snap.members == frozenset({"AAPL", "MSFT", "NVDA"})
    assert snap.trade_date == pd.Timestamp("2021-03-19")


# ── T7.2: future constituent not present in prior week snapshot ───────────────

def test_future_constituent_not_present_in_prior_week():
    """A ticker added Monday (asof=Monday 16:00) is NOT in Friday's snapshot."""
    store = _store_from_csv("""
        trade_date,ticker,asof_timestamp
        2021-03-19,AAPL,2021-03-19T16:00:00
        2021-03-19,MSFT,2021-03-19T16:00:00
        2021-03-22,TSLA,2021-03-22T16:00:00
    """)
    # Friday EOD asof: cannot see Monday's addition.
    snap = store.get_snapshot(
        pd.Timestamp("2021-03-19"), asof=pd.Timestamp("2021-03-19T23:59:59")
    )
    assert "TSLA" not in snap.members, "future constituent leaked into prior snapshot"
    assert snap.members == frozenset({"AAPL", "MSFT"})

    # Also verify: get_snapshot for 2021-03-22 with asof Monday 16:00 DOES see TSLA.
    # (TSLA's asof_timestamp is 2021-03-22T16:00, so it's visible at that asof.)
    snap_mon = store.get_snapshot(
        pd.Timestamp("2021-03-22"), asof=pd.Timestamp("2021-03-22T23:59:59")
    )
    assert "TSLA" in snap_mon.members


# ── T7.3: delisted ticker invisible after removal ─────────────────────────────

def test_delisted_ticker_not_in_future_snapshot():
    """After a ticker's last appearance, it does not appear in later snapshots.

    The store only returns members whose rows exist with asof_timestamp <= asof.
    If INTC has no row for 2021-03-26, the store raises DataNotAvailableError
    for that date — there is no implicit carry-forward of old memberships.
    """
    store = _store_from_csv("""
        trade_date,ticker,asof_timestamp
        2021-03-19,AAPL,2021-03-19T16:00:00
        2021-03-19,INTC,2021-03-19T16:00:00
        2021-03-26,AAPL,2021-03-26T16:00:00
    """)
    # INTC has no row for 2021-03-26 → not in that snapshot.
    snap = store.get_snapshot(
        pd.Timestamp("2021-03-26"), asof=pd.Timestamp("2021-03-26T23:59:59")
    )
    assert "INTC" not in snap.members, "delisted ticker appeared in later snapshot"
    assert "AAPL" in snap.members


def test_delisted_ticker_absent_after_delist_date():
    """A delisted ticker is absent from later as-of-visible snapshots."""
    store = _store_from_csv("""
        trade_date,ticker,asof_timestamp
        2021-06-04,KEEP,2021-06-04T16:00:00
        2021-06-04,OLD,2021-06-04T16:00:00
        2021-06-07,KEEP,2021-06-07T16:00:00
    """)

    before = store.get_snapshot(
        pd.Timestamp("2021-06-04"), asof=pd.Timestamp("2021-06-04T16:00:00")
    )
    after = store.get_snapshot(
        pd.Timestamp("2021-06-07"), asof=pd.Timestamp("2021-06-07T16:00:00")
    )

    assert "OLD" in before.members
    assert "OLD" not in after.members
    assert after.asof_timestamp == pd.Timestamp("2021-06-07T16:00:00")


def test_merged_ticker_disappears_on_merge_date():
    """Merger disappearance does not imply substitution into the acquirer."""
    store = _store_from_csv("""
        trade_date,ticker,asof_timestamp
        2021-07-09,ACQA,2021-07-09T16:00:00
        2021-07-09,MERG,2021-07-09T16:00:00
        2021-07-12,ACQA,2021-07-12T16:00:00
    """)

    pre_merge = store.get_snapshot(
        pd.Timestamp("2021-07-09"), asof=pd.Timestamp("2021-07-09T16:00:00")
    )
    post_merge = store.get_snapshot(
        pd.Timestamp("2021-07-12"), asof=pd.Timestamp("2021-07-12T16:00:00")
    )

    assert "MERG" in pre_merge.members
    assert "MERG" not in post_merge.members
    assert post_merge.members == frozenset({"ACQA"})


def test_renamed_ticker_not_in_old_symbol_after_rename():
    """Old and new rename symbols are independent; no bridge is inferred."""
    store = _store_from_csv("""
        trade_date,ticker,asof_timestamp
        2021-08-13,STAL,2021-08-13T16:00:00
        2021-08-16,FRESH,2021-08-16T16:00:00
    """)

    old_symbol_snapshot = store.get_snapshot(
        pd.Timestamp("2021-08-13"), asof=pd.Timestamp("2021-08-13T16:00:00")
    )
    new_symbol_snapshot = store.get_snapshot(
        pd.Timestamp("2021-08-16"), asof=pd.Timestamp("2021-08-16T16:00:00")
    )

    assert old_symbol_snapshot.members == frozenset({"STAL"})
    assert "STAL" not in new_symbol_snapshot.members
    assert new_symbol_snapshot.members == frozenset({"FRESH"})


# ── T7.4: missing trade date raises DataNotAvailableError ─────────────────────

def test_missing_trade_date_raises():
    """DataNotAvailableError raised when no rows match the requested trade_date."""
    store = _store_from_csv("""
        trade_date,ticker,asof_timestamp
        2021-03-19,AAPL,2021-03-19T16:00:00
    """)
    with pytest.raises(DataNotAvailableError):
        store.get_snapshot(
            pd.Timestamp("2021-03-26"), asof=pd.Timestamp("2021-03-26T23:59:59")
        )


# ── T7.5: strict asof filter — data published after asof is invisible ─────────

def test_strict_asof_filter():
    """Rows with asof_timestamp > asof are invisible even on the correct trade_date."""
    store = _store_from_csv("""
        trade_date,ticker,asof_timestamp
        2021-03-19,AAPL,2021-03-19T16:00:00
        2021-03-19,MSFT,2021-03-20T09:00:00
    """)
    # Querying with asof=Friday 16:00: MSFT (published Saturday 9am) is NOT visible.
    snap = store.get_snapshot(
        pd.Timestamp("2021-03-19"), asof=pd.Timestamp("2021-03-19T16:00:00")
    )
    assert "AAPL" in snap.members
    assert "MSFT" not in snap.members, "late-published row leaked through strict asof filter"

    # With asof after MSFT's publish time, it IS visible.
    snap2 = store.get_snapshot(
        pd.Timestamp("2021-03-19"), asof=pd.Timestamp("2021-03-20T23:59:59")
    )
    assert "MSFT" in snap2.members
