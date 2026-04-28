"""Fail-closed constituent membership data contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


class CsvConstituentStore(ConstituentStore):
    """CSV-backed constituent store with strict as-of semantics.

    CSV format (one row per ticker per trade_date):
        trade_date,ticker,asof_timestamp
        2021-01-04,AAPL,2021-01-04T16:00:00
        2021-01-04,MSFT,2021-01-04T16:00:00

    as-of rule: only rows where asof_timestamp <= asof are visible.
    A constituent added Monday (asof=Monday 16:00) is NOT visible to a
    Friday-EOD decision (asof=Friday 16:00) that precedes it.
    """

    def __init__(self, path: Path) -> None:
        df = pd.read_csv(path)
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
        df["asof_timestamp"] = pd.to_datetime(df["asof_timestamp"], utc=False)
        df["ticker"] = df["ticker"].str.strip().str.upper()
        self._df = df

    def get_snapshot(
        self, trade_date: pd.Timestamp, asof: pd.Timestamp
    ) -> PITConstituentSnapshot:
        """Return the constituent membership visible as of `asof` on `trade_date`.

        Raises DataNotAvailableError if no rows match.
        """
        trade_date = pd.Timestamp(trade_date).normalize()
        asof = pd.Timestamp(asof)
        mask = (self._df["trade_date"] == trade_date) & (self._df["asof_timestamp"] <= asof)
        rows = self._df.loc[mask]
        if rows.empty:
            raise DataNotAvailableError(
                f"no constituent data for trade_date={trade_date.date()} asof={asof}"
            )
        members = frozenset(rows["ticker"].tolist())
        latest_asof = pd.Timestamp(rows["asof_timestamp"].max())
        return PITConstituentSnapshot(
            trade_date=trade_date,
            members=members,
            asof_timestamp=latest_asof,
        )
