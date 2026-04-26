"""Fail-closed QQQ holdings weight data contract."""

from __future__ import annotations

import pandas as pd

from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError


class WeightStore:
    """Interface for point-in-time holdings weights."""

    def get_weights(self, trade_date: pd.Timestamp, asof: pd.Timestamp) -> dict[str, float]:
        del trade_date, asof
        raise DataNotAvailableError("weight store is not configured")
