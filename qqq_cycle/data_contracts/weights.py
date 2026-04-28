"""Fail-closed QQQ holdings weight data contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError


class WeightStore:
    """Interface for point-in-time holdings weights."""

    def get_weights(self, trade_date: pd.Timestamp, asof: pd.Timestamp) -> dict[str, float]:
        del trade_date, asof
        raise DataNotAvailableError("weight store is not configured")


class CsvWeightStore(WeightStore):
    """CSV-backed weight store with strict as-of semantics.

    CSV format:
        trade_date,ticker,weight,asof_timestamp
        2021-01-04,AAPL,0.115,2021-01-04T16:00:00
        2021-01-04,MSFT,0.097,2021-01-04T16:00:00

    as-of rule: only rows where asof_timestamp <= asof are visible.
    """

    def __init__(self, path: Path) -> None:
        df = pd.read_csv(path)
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
        df["asof_timestamp"] = pd.to_datetime(df["asof_timestamp"], utc=False)
        df["ticker"] = df["ticker"].str.strip().str.upper()
        df["weight"] = df["weight"].astype(float)
        self._df = df

    def get_weights(self, trade_date: pd.Timestamp, asof: pd.Timestamp) -> dict[str, float]:
        """Return {ticker: weight} visible as of `asof` on `trade_date`.

        Raises DataNotAvailableError if no rows match.
        """
        trade_date = pd.Timestamp(trade_date).normalize()
        asof = pd.Timestamp(asof)
        mask = (self._df["trade_date"] == trade_date) & (self._df["asof_timestamp"] <= asof)
        rows = self._df.loc[mask]
        if rows.empty:
            raise DataNotAvailableError(
                f"no weight data for trade_date={trade_date.date()} asof={asof}"
            )
        return dict(zip(rows["ticker"], rows["weight"]))
