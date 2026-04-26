"""Fail-closed raw price data contract."""

from __future__ import annotations

import pandas as pd

from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError


class RawPriceStore:
    """Interface for raw closes with source/as-of timestamps.

    Production implementations must return unadjusted closes only. This base
    store fails closed to prevent accidental hindsight-adjusted use.
    """

    def get_raw_close(self, ticker: str, trade_date: pd.Timestamp, asof: pd.Timestamp) -> float:
        del ticker, trade_date, asof
        raise DataNotAvailableError("raw price store is not configured")
