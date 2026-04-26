"""PIT adjusted-close contract interfaces and degraded-mode behavior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class PITDataError(RuntimeError):
    """Base class for PIT adjusted-close contract failures."""


class DataNotAvailableError(PITDataError):
    """Raised when point-in-time corporate-action data is unavailable."""


class HindsightAdjustedDataError(DataNotAvailableError):
    """Raised when only hindsight/backward-adjusted data is available."""


class InsufficientHistoryError(DataNotAvailableError):
    """Raised when an adjusted PIT window cannot satisfy the requested length."""


@dataclass(frozen=True)
class PITPriceBar:
    """Single-day point-in-time adjusted price bar.

    `asof_timestamp` is the time at which the raw close and cumulative factors
    were knowable. It must be on or after `trade_date`.
    """

    trade_date: pd.Timestamp
    ticker: str
    raw_close: float
    split_factor_cum_pti: float
    dividend_factor_cum_pti: float
    adj_close_pti: float
    asof_timestamp: pd.Timestamp

    def __post_init__(self) -> None:
        trade = pd.Timestamp(self.trade_date)
        asof = pd.Timestamp(self.asof_timestamp)
        if asof.normalize() < trade.normalize():
            raise ValueError("asof_timestamp must be on or after trade_date")
        expected = self.raw_close * self.split_factor_cum_pti * self.dividend_factor_cum_pti
        if not np.isclose(expected, self.adj_close_pti, rtol=0.0, atol=1e-8):
            raise ValueError("adj_close_pti must equal raw_close * split_factor * dividend_factor")


@dataclass(frozen=True)
class DegradedMode:
    """Micro/risk degradation decision when PIT data is unavailable."""

    micro_enabled: bool
    h_t: None
    rho_t: None
    reason: str


class PITAdjustmentEngine:
    """Interface for corporate-action adjusted prices with strict PIT semantics.

    This base class intentionally fails closed. A production implementation must
    override methods with data sourced from raw closes and corporate actions
    knowable on or before `asof`.
    """

    def __init__(self, *, only_hindsight_adjusted_available: bool = False) -> None:
        self.only_hindsight_adjusted_available = only_hindsight_adjusted_available

    def _fail_if_hindsight_only(self) -> None:
        if self.only_hindsight_adjusted_available:
            raise HindsightAdjustedDataError(
                "only hindsight/backward-adjusted close is available; PIT adjustment required"
            )

    def get_adj_close(
        self, ticker: str, trade_date: pd.Timestamp, asof: pd.Timestamp
    ) -> float:
        """Return single-day adjusted close knowable as of `asof`.

        Raises:
            HindsightAdjustedDataError if only backward-adjusted data exists.
            DataNotAvailableError if no PIT implementation is wired.
        """

        del ticker, trade_date, asof
        self._fail_if_hindsight_only()
        raise DataNotAvailableError("PIT adjusted-close engine is not implemented")

    def get_adjusted_window(
        self, ticker: str, end_date: pd.Timestamp, window: int, asof: pd.Timestamp
    ) -> pd.Series:
        """Return PIT-adjusted close window ending at `end_date`.

        Required scaling contract:
            P_adj(tau | asof) = P_raw(tau) * CUM_FAC(asof) / CUM_FAC(tau)

        All raw closes and corporate-action factors must be knowable on or
        before `asof`. Hindsight-adjusted vendor close is forbidden.
        """

        del ticker, end_date, window, asof
        self._fail_if_hindsight_only()
        raise DataNotAvailableError("PIT adjusted-window engine is not implemented")


def degrade_micro_mode(engine: PITAdjustmentEngine | None) -> DegradedMode:
    """Return lightweight-mode degradation when PIT adjusted prices are absent."""

    reason = "PIT adjustment engine unavailable"
    if engine is not None and engine.only_hindsight_adjusted_available:
        reason = "only hindsight-adjusted prices available"
    return DegradedMode(micro_enabled=False, h_t=None, rho_t=None, reason=reason)
