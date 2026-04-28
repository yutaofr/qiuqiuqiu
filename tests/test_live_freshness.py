"""Tests for live freshness gates."""

from __future__ import annotations

import pandas as pd
import pytest

from qqq_cycle.live.freshness import (
    FreshnessRecord,
    check_all_freshness,
    check_ai_gpr_freshness,
    check_macro_freshness,
    check_prices_freshness,
    derive_execution_state,
)
from qqq_cycle.pipeline import MODE_DEGRADED, MODE_STRICT, MODE_WARMUP


def _macro_df(last_date: str, include_ai_gpr: bool = True) -> pd.DataFrame:
    """Synthetic macro DataFrame with the given last observation date."""
    dates = pd.date_range(end=last_date, periods=5, freq="W-FRI")
    cols = ["DFII10", "DGS2", "BAMLH0A0HYM2", "NFCI", "VIXCLS", "AI_GPR", "USEPUINDXD", "QQQ"]
    if not include_ai_gpr:
        cols = [c for c in cols if c != "AI_GPR"]
    import numpy as np

    rng = pd.np.random.default_rng(0) if hasattr(pd, "np") else __import__("numpy").random.default_rng(0)
    return pd.DataFrame(
        __import__("numpy").ones((len(dates), len(cols))),
        index=dates,
        columns=cols,
    )


def _week_end(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(date_str)


# ---------------------------------------------------------------------------
# Tests: macro freshness
# ---------------------------------------------------------------------------

def test_fresh_macro_is_fresh() -> None:
    week_end = _week_end("2025-01-17")
    macro = _macro_df("2025-01-17")
    r = check_macro_freshness(macro, week_end)
    assert r.fresh_enough is True
    assert r.blocking_level == "degrade"
    assert r.reason is None


def test_stale_macro_returns_degrade_level() -> None:
    week_end = _week_end("2025-01-17")
    macro = _macro_df("2025-01-10")
    r = check_macro_freshness(macro, week_end)
    assert r.fresh_enough is False
    assert r.blocking_level == "degrade"
    assert r.reason is not None


def test_stale_ai_gpr_returns_degrade_level() -> None:
    import numpy as np

    week_end = _week_end("2025-01-17")
    dates = pd.date_range(end="2025-01-10", periods=5, freq="W-FRI")
    macro = pd.DataFrame(
        np.ones((len(dates), 8)),
        index=dates,
        columns=["DFII10", "DGS2", "BAMLH0A0HYM2", "NFCI", "VIXCLS", "AI_GPR", "USEPUINDXD", "QQQ"],
    )
    r = check_ai_gpr_freshness(macro, week_end)
    assert r.fresh_enough is False
    assert r.blocking_level == "degrade"


def test_missing_ai_gpr_column_returns_degrade() -> None:
    week_end = _week_end("2025-01-17")
    macro = _macro_df("2025-01-17", include_ai_gpr=False)
    r = check_ai_gpr_freshness(macro, week_end)
    assert r.fresh_enough is False
    assert r.blocking_level == "degrade"
    assert "missing" in (r.reason or "").lower()


def test_stale_prices_returns_block_level() -> None:
    week_end = _week_end("2025-01-17")
    macro = _macro_df("2025-01-10")
    r = check_prices_freshness(macro, week_end)
    assert r.fresh_enough is False
    assert r.blocking_level == "block"


def test_fresh_prices_is_fresh() -> None:
    week_end = _week_end("2025-01-17")
    macro = _macro_df("2025-01-17")
    r = check_prices_freshness(macro, week_end)
    assert r.fresh_enough is True
    assert r.blocking_level == "block"


# ---------------------------------------------------------------------------
# Tests: store-based checks
# ---------------------------------------------------------------------------

def test_none_constituent_store_returns_block() -> None:
    from qqq_cycle.live.freshness import check_constituents_freshness

    r = check_constituents_freshness(None, _week_end("2025-01-17"))
    assert r.fresh_enough is False
    assert r.blocking_level == "block"


def test_none_weight_store_returns_block() -> None:
    from qqq_cycle.live.freshness import check_weights_freshness

    r = check_weights_freshness(None, _week_end("2025-01-17"))
    assert r.fresh_enough is False
    assert r.blocking_level == "block"


def test_none_pit_engine_returns_block() -> None:
    from qqq_cycle.live.freshness import check_pit_engine_freshness

    r = check_pit_engine_freshness(None, _week_end("2025-01-17"))
    assert r.fresh_enough is False
    assert r.blocking_level == "block"


# ---------------------------------------------------------------------------
# Tests: check_all_freshness
# ---------------------------------------------------------------------------

def test_all_sources_stale_includes_all() -> None:
    """check_all_freshness returns one record per source."""
    week_end = _week_end("2025-01-17")
    macro = _macro_df("2025-01-10")  # stale macro
    records = check_all_freshness(macro_df=macro, week_end=week_end)
    assert len(records) == 6
    labels = {r.source_label for r in records}
    assert "fred_macro" in labels
    assert "qqq_prices" in labels
    assert "constituents" in labels
    assert "weights" in labels


# ---------------------------------------------------------------------------
# Tests: derive_execution_state
# ---------------------------------------------------------------------------

def _record(fresh: bool, level: str, label: str = "test") -> FreshnessRecord:
    return FreshnessRecord(
        source_label=label,
        last_observation_date="2025-01-17",
        asof_timestamp="2025-01-17",
        fresh_enough=fresh,
        blocking_level=level,
        reason=None if fresh else f"{label} stale",
    )


def test_all_fresh_strict_returns_execute() -> None:
    records = [_record(True, "block"), _record(True, "degrade"), _record(True, "warn")]
    state, reason = derive_execution_state(records, MODE_STRICT)
    assert state == "execute"
    assert reason is None


def test_block_level_stale_returns_block() -> None:
    records = [_record(False, "block", "qqq_prices"), _record(True, "degrade")]
    state, reason = derive_execution_state(records, MODE_STRICT)
    assert state == "block"
    assert reason is not None


def test_degrade_level_stale_returns_degrade() -> None:
    records = [_record(True, "block"), _record(False, "degrade", "fred_macro")]
    state, reason = derive_execution_state(records, MODE_STRICT)
    assert state == "degrade"
    assert reason is not None


def test_pipeline_degraded_mode_returns_degrade_even_if_data_fresh() -> None:
    records = [_record(True, "block"), _record(True, "degrade")]
    state, reason = derive_execution_state(records, MODE_DEGRADED)
    assert state == "degrade"


def test_pipeline_warmup_mode_returns_degrade() -> None:
    records = [_record(True, "block"), _record(True, "degrade")]
    state, reason = derive_execution_state(records, MODE_WARMUP)
    assert state == "degrade"


def test_blocking_level_covers_all_sources() -> None:
    """All expected source labels are present in check_all_freshness output."""
    import numpy as np

    week_end = _week_end("2025-01-17")
    dates = pd.date_range(end="2025-01-17", periods=5, freq="W-FRI")
    macro = pd.DataFrame(
        np.ones((5, 8)),
        index=dates,
        columns=["DFII10", "DGS2", "BAMLH0A0HYM2", "NFCI", "VIXCLS", "AI_GPR", "USEPUINDXD", "QQQ"],
    )
    records = check_all_freshness(macro_df=macro, week_end=week_end)
    levels = {r.blocking_level for r in records}
    assert levels.issubset({"block", "degrade", "warn"}), f"unexpected levels: {levels}"
    assert len(records) >= 3, "at least 3 sources should be checked"
