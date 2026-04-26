"""Fail-closed raw price data contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError, HindsightAdjustedDataError


@dataclass(frozen=True)
class RawPriceObservation:
    """Point-in-time raw close observation."""

    trade_date: pd.Timestamp
    ticker: str
    raw_close: float
    asof_timestamp: pd.Timestamp
    source: str


class RawPriceStore:
    """Interface for raw closes with source/as-of timestamps.

    Production implementations must return unadjusted closes only. This base
    store fails closed to prevent accidental hindsight-adjusted use.
    """

    def get_raw_close(self, ticker: str, trade_date: pd.Timestamp, asof: pd.Timestamp) -> float:
        del ticker, trade_date, asof
        raise DataNotAvailableError("raw price store is not configured")


class CsvRawPriceStore(RawPriceStore):
    """CSV-backed raw close store for audited local fixtures/imports.

    Required columns: `trade_date`, `ticker`, `raw_close`, `asof_timestamp`.
    Forbidden columns include adjusted-close variants. Rows are visible only
    when `asof_timestamp <= asof`.
    """

    REQUIRED = {"trade_date", "ticker", "raw_close", "asof_timestamp"}
    FORBIDDEN = {"adjusted_close", "adj_close", "adj close", "close_adjusted"}

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        raw = pd.read_csv(self.path)
        normalized = {str(col).strip().lower(): col for col in raw.columns}
        forbidden = self.FORBIDDEN.intersection(normalized)
        if forbidden:
            raise HindsightAdjustedDataError(
                f"raw price CSV contains forbidden adjusted columns: {sorted(forbidden)}"
            )
        missing = self.REQUIRED.difference(normalized)
        if missing:
            raise DataNotAvailableError(
                f"raw price CSV missing required columns: {sorted(missing)}"
            )
        frame = raw.rename(columns={original: key for key, original in normalized.items()})
        self._frame = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(frame["trade_date"]),
                "ticker": frame["ticker"].astype(str),
                "raw_close": pd.to_numeric(frame["raw_close"], errors="coerce"),
                "asof_timestamp": pd.to_datetime(frame["asof_timestamp"]),
                "source": frame["source"].astype(str) if "source" in frame else self.path.name,
            }
        ).dropna(subset=["trade_date", "ticker", "raw_close", "asof_timestamp"])
        self._frame = self._frame.sort_values(["ticker", "trade_date", "asof_timestamp"])

    def get_raw_close(self, ticker: str, trade_date: pd.Timestamp, asof: pd.Timestamp) -> float:
        trade_ts = pd.Timestamp(trade_date)
        asof_ts = pd.Timestamp(asof)
        rows = self._frame[
            (self._frame["ticker"] == ticker)
            & (self._frame["trade_date"] == trade_ts)
            & (self._frame["asof_timestamp"] <= asof_ts)
        ]
        if rows.empty:
            raise DataNotAvailableError(
                f"no raw close visible for {ticker} trade_date={trade_ts} asof={asof_ts}"
            )
        return float(rows.iloc[-1]["raw_close"])

    def to_series(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        rows = self._frame[
            (self._frame["ticker"] == ticker)
            & (self._frame["trade_date"] >= pd.Timestamp(start))
            & (self._frame["trade_date"] <= pd.Timestamp(end))
        ]
        rows = rows.groupby("trade_date", as_index=False).tail(1).sort_values("trade_date")
        return pd.Series(
            rows["raw_close"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(rows["trade_date"]),
            name=ticker,
        )
