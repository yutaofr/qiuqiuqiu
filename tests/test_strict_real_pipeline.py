"""T10: Strict real pipeline end-to-end tests using seeded micro data.

All tests skip if cache/micro/constituents.csv does not exist.
Run scripts/seed_micro_data.py to generate it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from qqq_cycle.data_contracts.constituents import CsvConstituentStore
from qqq_cycle.data_contracts.pit_adjustment import (
    CsvPITAdjustmentEngine,
    DataNotAvailableError,
    PITAdjustmentEngine,
)
from qqq_cycle.data_contracts.weights import CsvWeightStore
from qqq_cycle.pipeline import (
    MODE_DEGRADED,
    MODE_STRICT,
    MODE_WARMUP,
    PipelineContracts,
    run_pipeline,
)

_MICRO_DIR = Path("cache/micro")
_REAL_CSV = Path("cache/real_replay/staging/weekly_inputs.csv")
_SKIP_MSG = "micro seed data not found; run scripts/seed_micro_data.py"


def _skip_if_no_micro():
    return pytest.mark.skipif(
        not (_MICRO_DIR / "constituents.csv").exists(), reason=_SKIP_MSG
    )


@pytest.fixture(scope="module")
def real_strict_results():
    """Run the strict real pipeline once and share results across tests."""
    inputs = pd.read_csv(_REAL_CSV, index_col=0, parse_dates=True)
    inputs.index = pd.to_datetime(inputs.index)
    pit = CsvPITAdjustmentEngine(_MICRO_DIR / "prices")
    cs = CsvConstituentStore(_MICRO_DIR / "constituents.csv")
    ws = CsvWeightStore(_MICRO_DIR / "weights.csv")
    contracts = PipelineContracts(
        pit_engine=pit,
        constituent_store=cs,
        weight_store=ws,
    )
    return run_pipeline(inputs, contracts=contracts)


# ── T10.1: at least one post-warmup week has non-null h_t and rho_t ──────────

@_skip_if_no_micro()
def test_real_strict_path_produces_nonnull_h_t(real_strict_results):
    """With real stores, at least one post-warmup week has h_t and rho_t non-null."""
    strict = [r for r in real_strict_results if r.mode == MODE_STRICT]
    assert len(strict) > 0, (
        "no strict rows produced — micro data coverage may not overlap with "
        "post-warmup period or z_wrob_156 window is not yet satisfied"
    )
    assert all(r.h_t is not None for r in strict), "strict row has null h_t"
    assert all(r.rho_t is not None for r in strict), "strict row has null rho_t"


# ── T10.2: strict_contracts_satisfied=True for real strict rows ──────────────

@_skip_if_no_micro()
def test_strict_contracts_satisfied_true(real_strict_results):
    """Real strict rows must have strict_contracts_satisfied=True."""
    strict = [r for r in real_strict_results if r.mode == MODE_STRICT]
    pytest.skip("requires at least one strict row") if not strict else None
    for r in strict:
        assert r.strict_contracts_satisfied is True, (
            f"strict row {r.week_end} has strict_contracts_satisfied={r.strict_contracts_satisfied}"
        )


# ── T10.3: degraded rows have non-empty degraded_reason ──────────────────────

@_skip_if_no_micro()
def test_degraded_rows_have_reason(real_strict_results):
    """Every degraded row in the real strict run has a non-empty degraded_reason."""
    degraded = [r for r in real_strict_results if r.mode == MODE_DEGRADED]
    for r in degraded:
        assert r.degraded_reason is not None and len(r.degraded_reason) > 0, (
            f"degraded row {r.week_end} has empty degraded_reason"
        )


# ── T10.4: no lookahead in weekly cutoff ─────────────────────────────────────

@_skip_if_no_micro()
def test_no_lookahead_in_weekly_cutoff():
    """Friday decision at week_end T does not consume Monday rebalance data.

    Creates an in-memory store where a new constituent is added on Monday.
    Verifies that the Friday snapshot (asof=Friday 23:59) does NOT include it.
    """
    import io
    import textwrap
    from qqq_cycle.data_contracts.constituents import CsvConstituentStore
    from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError

    csv = textwrap.dedent("""
        trade_date,ticker,asof_timestamp
        2021-03-19,AAPL,2021-03-19T16:00:00
        2021-03-19,MSFT,2021-03-19T16:00:00
        2021-03-22,AAPL,2021-03-22T16:00:00
        2021-03-22,MSFT,2021-03-22T16:00:00
        2021-03-22,TSLA,2021-03-22T16:00:00
    """).strip()

    df = pd.read_csv(io.StringIO(csv))
    store = CsvConstituentStore.__new__(CsvConstituentStore)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    df["asof_timestamp"] = pd.to_datetime(df["asof_timestamp"], utc=False)
    df["ticker"] = df["ticker"].str.strip().str.upper()
    store._df = df

    friday = pd.Timestamp("2021-03-19")
    friday_eod = friday + pd.Timedelta(hours=23, minutes=59, seconds=59)

    snap_friday = store.get_snapshot(friday, asof=friday_eod)
    assert "TSLA" not in snap_friday.members, (
        "TSLA (added Monday) leaked into Friday snapshot — lookahead violation"
    )

    monday = pd.Timestamp("2021-03-22")
    monday_eod = monday + pd.Timedelta(hours=23, minutes=59, seconds=59)
    snap_monday = store.get_snapshot(monday, asof=monday_eod)
    assert "TSLA" in snap_monday.members


# ── T10.5: pipeline handles missing ticker price gracefully ──────────────────

@_skip_if_no_micro()
def test_degraded_when_prices_missing_for_ticker():
    """Pipeline degrades gracefully when constituent has no price data.

    We create contracts where the PIT engine is an InMemoryPITAdjustmentEngine
    with no price data.  The micro loop will fail to compute h_t for every week
    (MicroLayerUnavailableError caught internally), producing NaN h_t.
    All post-warmup rows must be DEGRADED, not raise an exception.
    """
    class _EmptyPITEngine(PITAdjustmentEngine):
        def get_adjusted_window(self, ticker, end_date, window, asof):
            raise DataNotAvailableError(f"no data for {ticker}")

        def get_adj_close(self, ticker, trade_date, asof):
            raise DataNotAvailableError(f"no data for {ticker}")

    inputs = pd.read_csv(_REAL_CSV, index_col=0, parse_dates=True)
    inputs.index = pd.to_datetime(inputs.index)

    cs = CsvConstituentStore(_MICRO_DIR / "constituents.csv")
    ws = CsvWeightStore(_MICRO_DIR / "weights.csv")
    contracts = PipelineContracts(
        pit_engine=_EmptyPITEngine(),
        constituent_store=cs,
        weight_store=ws,
    )
    results = run_pipeline(inputs, contracts=contracts)
    post_warmup = [r for r in results if r.mode != MODE_WARMUP]
    assert len(post_warmup) > 0
    assert all(r.h_t is None for r in post_warmup), (
        "h_t must be null when PIT engine has no data"
    )
    assert all(r.rho_t is None for r in post_warmup), (
        "rho_t must be null when PIT engine has no data"
    )
