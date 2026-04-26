"""Fail-closed constituent membership data contract."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError


@dataclass(frozen=True)
class PITConstituentSnapshot:
    """Point-in-time constituent membership snapshot."""

    trade_date: pd.Timestamp
    members: frozenset[str]
    asof_timestamp: pd.Timestamp


class ConstituentStore:
    """Interface for point-in-time constituent snapshots."""

    def get_snapshot(
        self, trade_date: pd.Timestamp, asof: pd.Timestamp
    ) -> PITConstituentSnapshot:
        del trade_date, asof
        raise DataNotAvailableError("constituent store is not configured")
