"""PIT adjusted-close contract interfaces and degraded-mode behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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

    source_label: str = "abstract"
    asof_semantics: str = "strict_pit"

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

        Inputs:
            ticker: Security identifier to retrieve.
            trade_date: Trading date of the raw close.
            asof: Decision timestamp; only corporate actions knowable on or
                before this timestamp may affect the returned value.

        Output:
            PIT-adjusted close for `ticker` and `trade_date`, expressed on the
            corporate-action basis visible at `asof`.

        Time semantics:
            Implementations must use raw close and corporate-action factors
            available as of `asof`; hindsight/backward-adjusted vendor series
            are forbidden for production strict mode.

        Failure modes:
            DataNotAvailableError: no PIT data source is wired or no row is
                visible at the requested timestamp.
            HindsightAdjustedDataError: only backward-adjusted data exists.
            InsufficientHistoryError: fewer than the requested `window` trading
                days are available for rolling-window callers.

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


class _RawPriceStoreProtocol(Protocol):
    def get_raw_close(self, ticker: str, trade_date: pd.Timestamp, asof: pd.Timestamp) -> float:
        ...


class _CorporateActionStoreProtocol(Protocol):
    def get_adjustment_factor(
        self,
        ticker: str,
        start_exclusive: pd.Timestamp,
        end_inclusive: pd.Timestamp,
        asof: pd.Timestamp,
    ) -> float:
        ...


class _IdentityResolverProtocol(Protocol):
    def resolve_symbol(self, ticker: str, trade_date: pd.Timestamp, asof: pd.Timestamp) -> str:
        ...


class LedgerPITAdjustmentEngine(PITAdjustmentEngine):
    """Production strict PIT engine backed by raw prices and normalized actions.

    Inputs:
        raw_price_store: Store that exposes only raw, unadjusted closes.
        corporate_action_store: Store of upstream-normalized equivalent factors.
        identity_resolver: Optional pure-rename resolver. It is required when a
            requested ticker's historical window crosses an explicit rename.

    Output/time semantics:
        Adjusted prices are reconstructed as:

            P_adj(tau) = P_raw(tau) * product(f_u for tau < u <= basis_date)

        using only raw closes and corporate-action records visible at `asof`.
        The engine never reads hindsight/backward-adjusted close series.

    Failure modes:
        DataNotAvailableError: missing raw close, missing visible identity for
            a rename boundary, or unavailable action ledger.
        InsufficientHistoryError: fewer than the requested window rows can be
            reconstructed.
    """

    source_label: str = "ledger_raw_close_corp_actions"
    asof_semantics: str = "strict_pit"

    def __init__(
        self,
        *,
        raw_price_store: _RawPriceStoreProtocol,
        corporate_action_store: _CorporateActionStoreProtocol,
        identity_resolver: _IdentityResolverProtocol | None = None,
    ) -> None:
        super().__init__(only_hindsight_adjusted_available=False)
        self.raw_price_store = raw_price_store
        self.corporate_action_store = corporate_action_store
        self.identity_resolver = identity_resolver

    def _resolve_symbol_for_date(
        self, ticker: str, trade_date: pd.Timestamp, asof: pd.Timestamp
    ) -> str:
        requested = ticker.strip().upper()
        if self.identity_resolver is None:
            return requested
        return self.identity_resolver.resolve_symbol(requested, trade_date, asof)

    def _raw_close_with_identity(
        self, ticker: str, trade_date: pd.Timestamp, asof: pd.Timestamp
    ) -> tuple[str, float]:
        requested = ticker.strip().upper()
        trade_ts = pd.Timestamp(trade_date).normalize()
        asof_ts = pd.Timestamp(asof)
        try:
            return requested, self.raw_price_store.get_raw_close(requested, trade_ts, asof_ts)
        except DataNotAvailableError as direct_exc:
            resolved = self._resolve_symbol_for_date(requested, trade_ts, asof_ts)
            if resolved == requested:
                raise direct_exc
            return resolved, self.raw_price_store.get_raw_close(resolved, trade_ts, asof_ts)

    def _adjusted_close_on_basis(
        self,
        ticker: str,
        trade_date: pd.Timestamp,
        basis_date: pd.Timestamp,
        asof: pd.Timestamp,
    ) -> float:
        symbol, raw_close = self._raw_close_with_identity(ticker, trade_date, asof)
        factor = self.corporate_action_store.get_adjustment_factor(
            symbol,
            start_exclusive=pd.Timestamp(trade_date).normalize(),
            end_inclusive=pd.Timestamp(basis_date).normalize(),
            asof=pd.Timestamp(asof),
        )
        return float(raw_close) * float(factor)

    def get_adj_close(
        self, ticker: str, trade_date: pd.Timestamp, asof: pd.Timestamp
    ) -> float:
        trade_ts = pd.Timestamp(trade_date).normalize()
        asof_ts = pd.Timestamp(asof)
        basis_date = asof_ts.normalize()
        return self._adjusted_close_on_basis(ticker, trade_ts, basis_date, asof_ts)

    def get_adjusted_window(
        self, ticker: str, end_date: pd.Timestamp, window: int, asof: pd.Timestamp
    ) -> pd.Series:
        if window < 1:
            raise ValueError("window must be >= 1")
        end_ts = pd.Timestamp(end_date).normalize()
        asof_ts = pd.Timestamp(asof)
        values: list[float] = []
        dates: list[pd.Timestamp] = []
        if hasattr(self.raw_price_store, "available_trade_dates"):
            candidate_dates = list(
                self.raw_price_store.available_trade_dates(ticker, end_ts, asof_ts)
            )
            if self.identity_resolver is not None:
                historical_probe = pd.bdate_range(end=end_ts, periods=window)[0]
                try:
                    historical_symbol = self.identity_resolver.resolve_symbol(
                        ticker, historical_probe, asof_ts
                    )
                except DataNotAvailableError:
                    historical_symbol = ticker
                if historical_symbol != ticker and hasattr(self.raw_price_store, "available_trade_dates"):
                    old_dates = list(
                        self.raw_price_store.available_trade_dates(
                            historical_symbol, end_ts, asof_ts
                        )
                    )
                    candidate_dates = sorted(set(candidate_dates) | set(old_dates))
            iterator = reversed(candidate_dates)
        else:
            lookback_days = max(window * 3, window + 10)
            iterator = reversed(pd.bdate_range(end=end_ts, periods=lookback_days))
        last_error: DataNotAvailableError | None = None
        for day in iterator:
            trade_ts = pd.Timestamp(day).normalize()
            try:
                values.append(
                    self._adjusted_close_on_basis(ticker, trade_ts, end_ts, asof_ts)
                )
                dates.append(trade_ts)
                if len(values) == window:
                    break
            except DataNotAvailableError as exc:
                last_error = exc
                continue
        if len(values) < window:
            raise InsufficientHistoryError(
                f"need {window} PIT rows for {ticker} ending {end_ts}; got {len(values)}"
            ) from last_error
        values.reverse()
        dates.reverse()
        return pd.Series(values, index=pd.DatetimeIndex(dates), name=ticker)


class InMemoryPITAdjustmentEngine(PITAdjustmentEngine):
    """Deterministic PIT adjustment engine for fixture and contract tests.

    This is not a production data loader. It operates only on supplied
    `PITPriceBar` records and filters every lookup by `asof_timestamp <= asof`.
    """

    source_label: str = "in_memory_fixture"

    def __init__(self, bars: list[PITPriceBar]) -> None:
        super().__init__(only_hindsight_adjusted_available=False)
        self._bars = pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp(bar.trade_date),
                    "ticker": bar.ticker,
                    "raw_close": float(bar.raw_close),
                    "cum_factor": float(
                        bar.split_factor_cum_pti * bar.dividend_factor_cum_pti
                    ),
                    "asof_timestamp": pd.Timestamp(bar.asof_timestamp),
                }
                for bar in bars
            ]
        )
        if self._bars.empty:
            self._bars = pd.DataFrame(
                columns=["trade_date", "ticker", "raw_close", "cum_factor", "asof_timestamp"]
            )
        self._bars = self._bars.sort_values(["ticker", "trade_date", "asof_timestamp"])

    def _visible_rows(self, ticker: str, asof: pd.Timestamp) -> pd.DataFrame:
        asof_ts = pd.Timestamp(asof)
        visible = self._bars[
            (self._bars["ticker"] == ticker) & (self._bars["asof_timestamp"] <= asof_ts)
        ]
        if visible.empty:
            raise DataNotAvailableError(f"no PIT rows visible for {ticker} as of {asof_ts}")
        return visible

    def _latest_visible_by_trade_date(self, ticker: str, asof: pd.Timestamp) -> pd.DataFrame:
        visible = self._visible_rows(ticker, asof)
        return visible.groupby("trade_date", as_index=False).tail(1).sort_values("trade_date")

    def get_adj_close(
        self, ticker: str, trade_date: pd.Timestamp, asof: pd.Timestamp
    ) -> float:
        rows = self._latest_visible_by_trade_date(ticker, asof)
        trade_ts = pd.Timestamp(trade_date)
        row = rows[rows["trade_date"] == trade_ts]
        if row.empty:
            raise DataNotAvailableError(
                f"no PIT close for {ticker} trade_date={trade_ts} as of {pd.Timestamp(asof)}"
            )
        rec = row.iloc[-1]
        return float(rec["raw_close"] * rec["cum_factor"])

    def get_adjusted_window(
        self, ticker: str, end_date: pd.Timestamp, window: int, asof: pd.Timestamp
    ) -> pd.Series:
        if window < 1:
            raise ValueError("window must be >= 1")
        rows = self._latest_visible_by_trade_date(ticker, asof)
        end_ts = pd.Timestamp(end_date)
        rows = rows[rows["trade_date"] <= end_ts].tail(window)
        if len(rows) < window:
            raise InsufficientHistoryError(
                f"need {window} PIT rows for {ticker} ending {end_ts}; got {len(rows)}"
            )
        basis = rows.iloc[-1]["cum_factor"]
        adjusted = rows["raw_close"].to_numpy(dtype=float) * (
            float(basis) / rows["cum_factor"].to_numpy(dtype=float)
        )
        return pd.Series(adjusted, index=pd.DatetimeIndex(rows["trade_date"]), name=ticker)


class CsvPITAdjustmentEngine(PITAdjustmentEngine):
    """CSV-backed PIT adjustment engine for Phase 7 backtesting.

    CSV format per ticker at prices_dir/{ticker}.csv:
        trade_date,raw_close,adj_close,asof_timestamp

    asof_timestamp = fetch date (conservative: all history appears known at
    fetch time). Not suitable for live production; documents limitations in
    seed_manifest.json.

    Relative-basis scaling:
        adj_price(tau) = raw_close(tau) * (adj_close[end] / raw_close[end])
    where [end] is the latest row visible as of asof.

    `asof_semantics` is `eod_same_day`: yfinance-derived CSV rows are
    retroactively adjusted at fetch time, which is a known production gap.
    """

    source_label: str = "csv_yfinance_eod"
    asof_semantics: str = "eod_same_day"

    def __init__(self, prices_dir: Path) -> None:
        super().__init__(only_hindsight_adjusted_available=False)
        self._prices_dir = Path(prices_dir)
        self._cache: dict[str, pd.DataFrame] = {}

    def _load_ticker(self, ticker: str) -> pd.DataFrame:
        if ticker not in self._cache:
            path = self._prices_dir / f"{ticker}.csv"
            if not path.exists():
                raise DataNotAvailableError(f"no price file for {ticker}: {path}")
            df = pd.read_csv(path)
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
            df["asof_timestamp"] = pd.to_datetime(df["asof_timestamp"], utc=False)
            df["raw_close"] = df["raw_close"].astype(float)
            df["adj_close"] = df["adj_close"].astype(float)
            self._cache[ticker] = df.sort_values("trade_date").reset_index(drop=True)
        return self._cache[ticker]

    def get_adj_close(
        self, ticker: str, trade_date: pd.Timestamp, asof: pd.Timestamp
    ) -> float:
        df = self._load_ticker(ticker)
        trade_ts = pd.Timestamp(trade_date).normalize()
        asof_ts = pd.Timestamp(asof)
        visible = df[(df["asof_timestamp"] <= asof_ts) & (df["trade_date"] == trade_ts)]
        if visible.empty:
            raise DataNotAvailableError(
                f"no PIT close for {ticker} date={trade_ts} asof={asof_ts}"
            )
        latest = visible.iloc[-1]
        basis = float(latest["adj_close"]) / float(latest["raw_close"])
        return float(latest["raw_close"]) * basis

    def get_adjusted_window(
        self, ticker: str, end_date: pd.Timestamp, window: int, asof: pd.Timestamp
    ) -> pd.Series:
        if window < 1:
            raise ValueError("window must be >= 1")
        df = self._load_ticker(ticker)
        end_ts = pd.Timestamp(end_date).normalize()
        asof_ts = pd.Timestamp(asof)
        visible = df[(df["asof_timestamp"] <= asof_ts) & (df["trade_date"] <= end_ts)]
        if visible.empty:
            raise DataNotAvailableError(
                f"no PIT rows for {ticker} end={end_ts} asof={asof_ts}"
            )
        rows = visible.tail(window)
        if len(rows) < window:
            raise InsufficientHistoryError(
                f"need {window} rows for {ticker} ending {end_ts}; got {len(rows)}"
            )
        latest = rows.iloc[-1]
        basis = float(latest["adj_close"]) / float(latest["raw_close"])
        adj_prices = rows["raw_close"].to_numpy(dtype=float) * basis
        return pd.Series(
            adj_prices, index=pd.DatetimeIndex(rows["trade_date"]), name=ticker
        )


def degrade_micro_mode(engine: object | None) -> DegradedMode:
    """Return lightweight-mode degradation when PIT adjusted prices are absent."""

    if engine is not None and (
        engine.__class__.__name__ in {"MacroMarketPriceContract", "CsvMacroMarketPriceStore"}
        or engine.__class__.__module__.endswith("macro_prices")
    ):
        raise DataNotAvailableError(
            "MacroMarketPriceContract is forbidden for micro-layer production paths"
        )
    reason = "PIT adjustment engine unavailable"
    if (
        isinstance(engine, PITAdjustmentEngine)
        and engine.only_hindsight_adjusted_available
    ):
        reason = "only hindsight-adjusted prices available"
    return DegradedMode(micro_enabled=False, h_t=None, rho_t=None, reason=reason)
