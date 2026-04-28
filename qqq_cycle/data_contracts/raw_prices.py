"""Fail-closed raw price data contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError, HindsightAdjustedDataError


@dataclass(frozen=True)
class RawPriceObservation:
    """Point-in-time raw close observation.

    Inputs:
        trade_date: Trading date of the unadjusted market close.
        ticker: Symbol under which the raw close was observed on trade_date.
        raw_close: Unadjusted close. Adjusted-close inputs are forbidden.
        source_label: Auditable source name for this observation.
        asof_timestamp: Timestamp at which the observation became knowable.
    """

    trade_date: pd.Timestamp
    ticker: str
    raw_close: float
    source_label: str
    asof_timestamp: pd.Timestamp


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
    Optional source column: `source_label` (or legacy `source`).
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
                "trade_date": pd.to_datetime(frame["trade_date"]).dt.normalize(),
                "ticker": frame["ticker"].astype(str).str.strip().str.upper(),
                "raw_close": pd.to_numeric(frame["raw_close"], errors="coerce"),
                "asof_timestamp": pd.to_datetime(frame["asof_timestamp"]),
                "source_label": (
                    frame["source_label"].astype(str)
                    if "source_label" in frame
                    else frame["source"].astype(str)
                    if "source" in frame
                    else self.path.name
                ),
            }
        ).dropna(subset=["trade_date", "ticker", "raw_close", "asof_timestamp"])
        self._frame = self._frame.sort_values(["ticker", "trade_date", "asof_timestamp"])
        self._lookup = {
            (str(ticker), pd.Timestamp(trade_date)): group.sort_values("asof_timestamp")
            for (ticker, trade_date), group in self._frame.groupby(["ticker", "trade_date"])
        }
        self._latest_raw_by_key = {
            key: (
                pd.Timestamp(group.iloc[-1]["asof_timestamp"]),
                float(group.iloc[-1]["raw_close"]),
            )
            for key, group in self._lookup.items()
        }
        self._dates_by_ticker = {
            str(ticker): pd.DatetimeIndex(group["trade_date"].drop_duplicates().sort_values())
            for ticker, group in self._frame.groupby("ticker")
        }

    def get_raw_close(self, ticker: str, trade_date: pd.Timestamp, asof: pd.Timestamp) -> float:
        trade_ts = pd.Timestamp(trade_date).normalize()
        asof_ts = pd.Timestamp(asof)
        rec = self._latest_raw_by_key.get((ticker.strip().upper(), trade_ts))
        if rec is None or rec[0] > asof_ts:
            raise DataNotAvailableError(
                f"no raw close visible for {ticker} trade_date={trade_ts} asof={asof_ts}"
            )
        return rec[1]

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

    def available_trade_dates(
        self, ticker: str, end_date: pd.Timestamp, asof: pd.Timestamp
    ) -> pd.DatetimeIndex:
        """Return raw-close trade dates visible as of `asof` through `end_date`."""

        end_ts = pd.Timestamp(end_date).normalize()
        asof_ts = pd.Timestamp(asof)
        dates = self._dates_by_ticker.get(ticker.strip().upper(), pd.DatetimeIndex([]))
        dates = dates[dates <= end_ts]
        symbol = ticker.strip().upper()
        visible = [date for date in dates if self._latest_raw_by_key[(symbol, pd.Timestamp(date))][0] <= asof_ts]
        return pd.DatetimeIndex(visible)


class InMemoryRawPriceStore(RawPriceStore):
    """Deterministic raw close store for PIT contract tests."""

    def __init__(self, observations: list[RawPriceObservation]) -> None:
        self._frame = pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp(obs.trade_date).normalize(),
                    "ticker": obs.ticker.strip().upper(),
                    "raw_close": float(obs.raw_close),
                    "source_label": obs.source_label,
                    "asof_timestamp": pd.Timestamp(obs.asof_timestamp),
                }
                for obs in observations
            ]
        )
        if self._frame.empty:
            self._frame = pd.DataFrame(
                columns=["trade_date", "ticker", "raw_close", "source_label", "asof_timestamp"]
            )
        self._frame = self._frame.sort_values(["ticker", "trade_date", "asof_timestamp"])
        self._lookup = {
            (str(ticker), pd.Timestamp(trade_date)): group.sort_values("asof_timestamp")
            for (ticker, trade_date), group in self._frame.groupby(["ticker", "trade_date"])
        }
        self._latest_raw_by_key = {
            key: (
                pd.Timestamp(group.iloc[-1]["asof_timestamp"]),
                float(group.iloc[-1]["raw_close"]),
            )
            for key, group in self._lookup.items()
        }
        self._dates_by_ticker = {
            str(ticker): pd.DatetimeIndex(group["trade_date"].drop_duplicates().sort_values())
            for ticker, group in self._frame.groupby("ticker")
        }

    def get_raw_close(self, ticker: str, trade_date: pd.Timestamp, asof: pd.Timestamp) -> float:
        trade_ts = pd.Timestamp(trade_date).normalize()
        asof_ts = pd.Timestamp(asof)
        rec = self._latest_raw_by_key.get((ticker.strip().upper(), trade_ts))
        if rec is None or rec[0] > asof_ts:
            raise DataNotAvailableError(
                f"no raw close visible for {ticker} trade_date={trade_ts} asof={asof_ts}"
            )
        return rec[1]

    def available_trade_dates(
        self, ticker: str, end_date: pd.Timestamp, asof: pd.Timestamp
    ) -> pd.DatetimeIndex:
        end_ts = pd.Timestamp(end_date).normalize()
        asof_ts = pd.Timestamp(asof)
        symbol = ticker.strip().upper()
        dates = self._dates_by_ticker.get(symbol, pd.DatetimeIndex([]))
        dates = dates[dates <= end_ts]
        visible = [date for date in dates if self._latest_raw_by_key[(symbol, pd.Timestamp(date))][0] <= asof_ts]
        return pd.DatetimeIndex(visible)
