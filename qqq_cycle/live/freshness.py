"""Data freshness gates for the live execution kernel.

Each check returns a FreshnessRecord describing whether a data source is fresh
enough for the decision week_end. Blocking levels:
    "block"  — execution must not proceed; position must not change
    "degrade" — execution permitted in degraded mode (macro/AI signals only)
    "warn"   — log a warning but allow full execution

Rule: data is fresh enough when last_observation_date >= decision week_end (Friday).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from qqq_cycle.data_contracts.constituents import ConstituentStore
    from qqq_cycle.data_contracts.weights import WeightStore
    from qqq_cycle.data_contracts.pit_adjustment import PITAdjustmentEngine
    from qqq_cycle.data_contracts.raw_prices import RawPriceStore


@dataclass(frozen=True)
class FreshnessRecord:
    """Freshness status for one data source as of a decision week."""

    source_label: str
    last_observation_date: str
    asof_timestamp: str
    fresh_enough: bool
    blocking_level: str  # "block" | "degrade" | "warn"
    reason: str | None


def _is_fresh(last_obs: pd.Timestamp | None, week_end: pd.Timestamp) -> bool:
    if last_obs is None:
        return False
    return pd.Timestamp(last_obs) >= week_end


def _ts_str(ts: pd.Timestamp | None) -> str:
    if ts is None:
        return "unknown"
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def check_macro_freshness(
    macro_df: pd.DataFrame,
    week_end: pd.Timestamp,
) -> FreshnessRecord:
    """Check FRED macro DataFrame (indexed by week-end dates)."""
    last_obs = macro_df.index.max() if len(macro_df) > 0 else None
    fresh = _is_fresh(last_obs, week_end)
    return FreshnessRecord(
        source_label="fred_macro",
        last_observation_date=_ts_str(last_obs),
        asof_timestamp=_ts_str(last_obs),
        fresh_enough=fresh,
        blocking_level="degrade",
        reason=None if fresh else f"macro last obs {_ts_str(last_obs)} < week_end {_ts_str(week_end)}",
    )


def check_ai_gpr_freshness(
    macro_df: pd.DataFrame,
    week_end: pd.Timestamp,
) -> FreshnessRecord:
    """Check AI-GPR column freshness (in the macro DataFrame)."""
    col = "AI_GPR"
    if col not in macro_df.columns:
        return FreshnessRecord(
            source_label="ai_gpr",
            last_observation_date="unknown",
            asof_timestamp="unknown",
            fresh_enough=False,
            blocking_level="degrade",
            reason="AI_GPR column missing from macro_df",
        )
    valid = macro_df[col].dropna()
    last_obs = valid.index.max() if len(valid) > 0 else None
    fresh = _is_fresh(last_obs, week_end)
    return FreshnessRecord(
        source_label="ai_gpr",
        last_observation_date=_ts_str(last_obs),
        asof_timestamp=_ts_str(last_obs),
        fresh_enough=fresh,
        blocking_level="degrade",
        reason=None if fresh else f"AI_GPR last obs {_ts_str(last_obs)} < week_end {_ts_str(week_end)}",
    )


def check_prices_freshness(
    macro_df: pd.DataFrame,
    week_end: pd.Timestamp,
) -> FreshnessRecord:
    """Check QQQ price freshness (in the macro DataFrame, used for state layer)."""
    col = "QQQ"
    if col not in macro_df.columns:
        return FreshnessRecord(
            source_label="qqq_prices",
            last_observation_date="unknown",
            asof_timestamp="unknown",
            fresh_enough=False,
            blocking_level="block",
            reason="QQQ column missing from macro_df",
        )
    valid = macro_df[col].dropna()
    last_obs = valid.index.max() if len(valid) > 0 else None
    fresh = _is_fresh(last_obs, week_end)
    return FreshnessRecord(
        source_label="qqq_prices",
        last_observation_date=_ts_str(last_obs),
        asof_timestamp=_ts_str(last_obs),
        fresh_enough=fresh,
        blocking_level="block",
        reason=None if fresh else f"QQQ last obs {_ts_str(last_obs)} < week_end {_ts_str(week_end)}",
    )


def check_constituents_freshness(
    constituent_store: "ConstituentStore | None",
    week_end: pd.Timestamp,
) -> FreshnessRecord:
    """Check constituent store freshness by probing for week_end snapshot."""
    from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError

    if constituent_store is None:
        return FreshnessRecord(
            source_label="constituents",
            last_observation_date="unknown",
            asof_timestamp="unknown",
            fresh_enough=False,
            blocking_level="block",
            reason="constituent_store not provided",
        )
    try:
        constituent_store.get_snapshot(week_end, asof=week_end)
        return FreshnessRecord(
            source_label="constituents",
            last_observation_date=_ts_str(week_end),
            asof_timestamp=_ts_str(week_end),
            fresh_enough=True,
            blocking_level="block",
            reason=None,
        )
    except DataNotAvailableError as exc:
        return FreshnessRecord(
            source_label="constituents",
            last_observation_date="unknown",
            asof_timestamp="unknown",
            fresh_enough=False,
            blocking_level="block",
            reason=f"constituent snapshot unavailable for {_ts_str(week_end)}: {exc}",
        )


def check_weights_freshness(
    weight_store: "WeightStore | None",
    week_end: pd.Timestamp,
) -> FreshnessRecord:
    """Check weight store freshness by probing for week_end weights."""
    from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError

    if weight_store is None:
        return FreshnessRecord(
            source_label="weights",
            last_observation_date="unknown",
            asof_timestamp="unknown",
            fresh_enough=False,
            blocking_level="block",
            reason="weight_store not provided",
        )
    try:
        weight_store.get_weights(week_end, asof=week_end)
        return FreshnessRecord(
            source_label="weights",
            last_observation_date=_ts_str(week_end),
            asof_timestamp=_ts_str(week_end),
            fresh_enough=True,
            blocking_level="block",
            reason=None,
        )
    except DataNotAvailableError as exc:
        return FreshnessRecord(
            source_label="weights",
            last_observation_date="unknown",
            asof_timestamp="unknown",
            fresh_enough=False,
            blocking_level="block",
            reason=f"weight snapshot unavailable for {_ts_str(week_end)}: {exc}",
        )


def check_pit_engine_freshness(
    pit_engine: "PITAdjustmentEngine | None",
    week_end: pd.Timestamp,
) -> FreshnessRecord:
    """Check PIT price engine availability (structural check only)."""
    if pit_engine is None:
        return FreshnessRecord(
            source_label="pit_prices",
            last_observation_date="unknown",
            asof_timestamp="unknown",
            fresh_enough=False,
            blocking_level="block",
            reason="pit_engine not provided",
        )
    return FreshnessRecord(
        source_label="pit_prices",
        last_observation_date=_ts_str(week_end),
        asof_timestamp=_ts_str(week_end),
        fresh_enough=True,
        blocking_level="block",
        reason=None,
    )


def check_all_freshness(
    *,
    macro_df: pd.DataFrame,
    week_end: pd.Timestamp,
    constituent_store: "ConstituentStore | None" = None,
    weight_store: "WeightStore | None" = None,
    pit_engine: "PITAdjustmentEngine | None" = None,
) -> list[FreshnessRecord]:
    """Run all freshness checks and return a list of FreshnessRecords."""
    return [
        check_macro_freshness(macro_df, week_end),
        check_ai_gpr_freshness(macro_df, week_end),
        check_prices_freshness(macro_df, week_end),
        check_constituents_freshness(constituent_store, week_end),
        check_weights_freshness(weight_store, week_end),
        check_pit_engine_freshness(pit_engine, week_end),
    ]


def derive_execution_state(
    freshness_records: list[FreshnessRecord],
    pipeline_mode: str,
) -> tuple[str, str | None]:
    """Return (execution_state, execution_block_reason) from freshness + pipeline mode.

    execution_state ∈ {"block", "degrade", "execute"}:
        block   — any block-level freshness failure
        degrade — any degrade-level failure OR pipeline mode != "strict"
        execute — all fresh and pipeline mode == "strict"
    """
    block_reasons = [
        r.reason
        for r in freshness_records
        if not r.fresh_enough and r.blocking_level == "block"
    ]
    if block_reasons:
        return "block", "; ".join(str(r) for r in block_reasons if r)

    degrade_reasons = [
        r.reason
        for r in freshness_records
        if not r.fresh_enough and r.blocking_level == "degrade"
    ]
    if degrade_reasons or pipeline_mode != "strict":
        reason = "; ".join(str(r) for r in degrade_reasons if r) or None
        return "degrade", reason

    return "execute", None
