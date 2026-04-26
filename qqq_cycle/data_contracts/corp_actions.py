"""Fail-closed corporate-action data contract."""

from __future__ import annotations

import pandas as pd

from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError


class CorporateActionStore:
    """Interface for PIT split/dividend cumulative factors."""

    def get_cumulative_factor(
        self, ticker: str, trade_date: pd.Timestamp, asof: pd.Timestamp
    ) -> float:
        del ticker, trade_date, asof
        raise DataNotAvailableError("corporate-action store is not configured")
