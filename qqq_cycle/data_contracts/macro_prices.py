"""Diagnostic macro-market price contract for state/stress replay only."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError


class PriceBasis(str, Enum):
    """Allowed macro price bases for diagnostic state/stress replay."""

    VENDOR_BACKWARD_ADJUSTED = "vendor_backward_adjusted"
    VENDOR_RAW_CLOSE = "vendor_raw_close"
    OFFICIAL_MARKET_CLOSE = "official_market_close"


ALLOWED_REPLAY_SCOPE = "state_stress_only"
FORBIDDEN_SCOPES = {"micro_layer", "production_h_t", "production_rho_t"}


@dataclass(frozen=True)
class MacroMarketPriceContract:
    """QQQ price observation for diagnostic state/stress replay.

    Input:
        trade_date: Market date for the close.
        ticker: Security ticker. Real replay currently consumes `QQQ`.
        close: Market close under `price_basis`.
        source_name: Vendor or official source label.
        fetch_timestamp: Timestamp when this diagnostic record was fetched.
        price_basis: One of `PriceBasis`.
    Output:
        Immutable validated price observation.
    Time semantics:
        This contract is not point-in-time adjusted and is therefore valid only
        for state/stress replay. It is forbidden for micro-layer production
        paths, production h_t, and production rho_t.
    """

    trade_date: pd.Timestamp
    ticker: str
    close: float
    source_name: str
    fetch_timestamp: pd.Timestamp
    price_basis: PriceBasis

    def __post_init__(self) -> None:
        trade = pd.Timestamp(self.trade_date)
        fetched = pd.Timestamp(self.fetch_timestamp)
        basis = PriceBasis(self.price_basis)
        if not self.ticker:
            raise ValueError("ticker is required")
        if not self.source_name:
            raise ValueError("source_name is required")
        if not np.isfinite(float(self.close)) or float(self.close) <= 0.0:
            raise ValueError("close must be finite and positive")
        object.__setattr__(self, "trade_date", trade)
        object.__setattr__(self, "fetch_timestamp", fetched)
        object.__setattr__(self, "price_basis", basis)


def require_macro_replay_scope(scope: str) -> None:
    """Raise unless a macro price contract is used for state/stress replay."""

    if scope != ALLOWED_REPLAY_SCOPE or scope in FORBIDDEN_SCOPES:
        raise DataNotAvailableError(
            "MacroMarketPriceContract is only allowed for state/stress replay"
        )


class CsvMacroMarketPriceStore:
    """CSV loader for diagnostic QQQ macro prices.

    Required columns: `trade_date`, `ticker`, `close`, `source_name`,
    `fetch_timestamp`, `price_basis`.
    """

    REQUIRED = {
        "trade_date",
        "ticker",
        "close",
        "source_name",
        "fetch_timestamp",
        "price_basis",
    }

    def __init__(
        self, path: str | Path, *, replay_scope: str = ALLOWED_REPLAY_SCOPE
    ) -> None:
        require_macro_replay_scope(replay_scope)
        self.path = Path(path)
        raw = pd.read_csv(self.path)
        normalized = {str(col).strip().lower(): col for col in raw.columns}
        missing = self.REQUIRED.difference(normalized)
        if missing:
            raise DataNotAvailableError(
                f"macro market price CSV missing required columns: {sorted(missing)}"
            )
        frame = raw.rename(columns={original: key for key, original in normalized.items()})
        basis = frame["price_basis"].map(lambda value: PriceBasis(str(value)))
        self._frame = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(frame["trade_date"]),
                "ticker": frame["ticker"].astype(str),
                "close": pd.to_numeric(frame["close"], errors="coerce"),
                "source_name": frame["source_name"].astype(str),
                "fetch_timestamp": pd.to_datetime(frame["fetch_timestamp"]),
                "price_basis": basis,
            }
        ).dropna(subset=["trade_date", "ticker", "close", "fetch_timestamp"])
        if self._frame.empty:
            raise DataNotAvailableError("macro market price CSV contains no usable rows")
        if (self._frame["close"] <= 0.0).any():
            raise ValueError("macro market price CSV contains non-positive close")
        unique_basis = set(self._frame["price_basis"])
        if len(unique_basis) != 1:
            raise ValueError("macro market price CSV must use one price_basis")
        self.price_basis = next(iter(unique_basis))
        self.source_names = tuple(sorted(set(self._frame["source_name"].astype(str))))
        self._frame = self._frame.sort_values(["ticker", "trade_date", "fetch_timestamp"])

    def to_series(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Return close series for `ticker` over `[start, end]`.

        Time semantics:
            This returns diagnostic market history for replay only. It does not
            expose PIT-adjusted windows and must not feed micro production code.
        """

        rows = self._frame[
            (self._frame["ticker"] == ticker)
            & (self._frame["trade_date"] >= pd.Timestamp(start))
            & (self._frame["trade_date"] <= pd.Timestamp(end))
        ]
        rows = rows.groupby("trade_date", as_index=False).tail(1).sort_values("trade_date")
        if rows.empty:
            raise DataNotAvailableError(f"no macro market prices for {ticker}")
        return pd.Series(
            rows["close"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(rows["trade_date"]),
            name=ticker,
        )
