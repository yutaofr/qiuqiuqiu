"""Machine-derived strict production epoch audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError
from qqq_cycle.data_contracts.weights import WeightSumViolationError, validate_weight_sum


@dataclass(frozen=True)
class StrictEpochManifest:
    """Strict epoch coverage result.

    `production_strict_epoch_start` is the first trading date from which every
    subsequent audited trading date satisfies PIT, constituent, weight, and
    identity continuity contracts. Dates before it are `degraded_by_design`,
    not production blockers.
    """

    production_strict_epoch_start: str | None
    production_strict_epoch_end: str | None
    constituent_coverage_ok: bool
    weight_coverage_ok: bool
    pit_coverage_ok: bool
    rename_identity_coverage_ok: bool
    open_blockers: list[str]
    row_modes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "production_strict_epoch_start": self.production_strict_epoch_start,
            "production_strict_epoch_end": self.production_strict_epoch_end,
            "constituent_coverage_ok": self.constituent_coverage_ok,
            "weight_coverage_ok": self.weight_coverage_ok,
            "pit_coverage_ok": self.pit_coverage_ok,
            "rename_identity_coverage_ok": self.rename_identity_coverage_ok,
            "open_blockers": list(self.open_blockers),
            "row_modes": dict(self.row_modes),
        }


def _asof_eod(day: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(day).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)


def _date_key(day: pd.Timestamp) -> str:
    return pd.Timestamp(day).strftime("%Y-%m-%d")


def _day_is_strict_eligible(
    day: pd.Timestamp,
    *,
    pit_engine: object,
    constituent_store: object,
    weight_store: object,
    pit_window: int,
) -> tuple[bool, set[str]]:
    blockers: set[str] = set()
    asof = _asof_eod(day)
    try:
        snapshot = constituent_store.get_snapshot(day, asof=asof)
    except DataNotAvailableError:
        return False, {"historical_constituent_coverage_incomplete"}
    try:
        weights = weight_store.get_weights(day, asof=asof)
        validate_weight_sum(weights)
    except (DataNotAvailableError, WeightSumViolationError):
        blockers.add("historical_weight_coverage_incomplete")
        weights = {}

    members = sorted(snapshot.members)
    weighted_members = [ticker for ticker in members if float(weights.get(ticker, 0.0)) > 0.0]
    if not weighted_members:
        blockers.add("historical_weight_coverage_incomplete")

    for ticker in weighted_members:
        try:
            pit_engine.get_adjusted_window(ticker, day, pit_window, asof=asof)
        except DataNotAvailableError:
            blockers.add("pit_coverage_incomplete")
            blockers.add("rename_identity_coverage_incomplete")
            break

    return not blockers, blockers


def derive_production_strict_epoch(
    trading_days: pd.DatetimeIndex,
    *,
    pit_engine: object,
    constituent_store: object,
    weight_store: object,
    pit_window: int = 60,
) -> StrictEpochManifest:
    """Derive the earliest production strict epoch from audited coverage.

    The function does not forward-fill or infer missing inputs. A candidate
    epoch is accepted only if every trading date from that candidate through
    the audit end satisfies all strict input contracts as of that date's EOD.
    """

    days = pd.DatetimeIndex(pd.to_datetime(trading_days)).sort_values()
    day_blockers: dict[pd.Timestamp, set[str]] = {}
    eligible: dict[pd.Timestamp, bool] = {}
    for day in days:
        ok, blockers = _day_is_strict_eligible(
            pd.Timestamp(day).normalize(),
            pit_engine=pit_engine,
            constituent_store=constituent_store,
            weight_store=weight_store,
            pit_window=pit_window,
        )
        eligible[pd.Timestamp(day).normalize()] = ok
        day_blockers[pd.Timestamp(day).normalize()] = blockers

    epoch: pd.Timestamp | None = None
    for i, day in enumerate(days):
        tail = [pd.Timestamp(d).normalize() for d in days[i:]]
        if all(eligible[d] for d in tail):
            epoch = pd.Timestamp(day).normalize()
            break

    if epoch is None:
        open_blockers = sorted(set().union(*day_blockers.values()) if day_blockers else set())
        return StrictEpochManifest(
            production_strict_epoch_start=None,
            production_strict_epoch_end=_date_key(days[-1]) if len(days) else None,
            constituent_coverage_ok="historical_constituent_coverage_incomplete" not in open_blockers,
            weight_coverage_ok="historical_weight_coverage_incomplete" not in open_blockers,
            pit_coverage_ok="pit_coverage_incomplete" not in open_blockers,
            rename_identity_coverage_ok="rename_identity_coverage_incomplete" not in open_blockers,
            open_blockers=open_blockers,
            row_modes={_date_key(day): "not_strict_eligible" for day in days},
        )

    row_modes = {
        _date_key(day): (
            "degraded_by_design"
            if pd.Timestamp(day).normalize() < epoch
            else "strict_eligible"
        )
        for day in days
    }
    return StrictEpochManifest(
        production_strict_epoch_start=_date_key(epoch),
        production_strict_epoch_end=_date_key(days[-1]) if len(days) else None,
        constituent_coverage_ok=True,
        weight_coverage_ok=True,
        pit_coverage_ok=True,
        rename_identity_coverage_ok=True,
        open_blockers=[],
        row_modes=row_modes,
    )
