"""Phase 11 no-lookahead weekly backtest engine.

The engine consumes weekly decisions and an execution-open price panel. At
period t it uses only the open attached to week_end[t] and week_end[t + 1] to
compute the held-period return.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qqq_cycle.portfolio.construction import BacktestConfig


PRICE_COLUMNS = ["exec_open_qqq", "exec_open_shy"]


def _flatten_yfinance_columns(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if not isinstance(raw.columns, pd.MultiIndex):
        if len(tickers) == 1:
            return raw.rename(columns={"Open": f"open_{tickers[0].lower()}"})
        raise RuntimeError("expected yfinance MultiIndex columns for multi-ticker download")

    frames: list[pd.Series] = []
    for ticker in tickers:
        series: pd.Series | None = None
        if ("Open", ticker) in raw.columns:
            series = raw[("Open", ticker)]
        elif (ticker, "Open") in raw.columns:
            series = raw[(ticker, "Open")]
        if series is None:
            raise RuntimeError(f"yfinance output missing Open for {ticker}")
        frames.append(series.rename(f"open_{ticker.lower()}"))
    return pd.concat(frames, axis=1)


def fetch_price_panel(
    start: str,
    end: str,
    cache_path: str | Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch or load QQQ/SHY adjusted opens aligned to weekly signals.

    Input:
        start, end: Inclusive signal week_end range.
        cache_path: CSV cache for the aligned execution panel.
        force_refresh: Re-download from yfinance when true.

    Output:
        DataFrame with week_end, exec_date, exec_open_qqq, exec_open_shy.

    Time/as-of semantics:
        Uses yfinance auto_adjust=True for total-return adjusted opens. Each
        week_end maps to the first trading day strictly after that Friday.
        The function is for Phase 11 research backtests; it does not provide a
        PIT micro-layer adjusted-close contract.
    """

    cache = Path(cache_path)
    if cache.exists() and not force_refresh:
        return pd.read_csv(cache, parse_dates=["week_end", "exec_date"])

    import yfinance as yf

    signal_start = pd.Timestamp(start)
    signal_end = pd.Timestamp(end)
    fetch_end = signal_end + pd.DateOffset(days=14)
    raw = yf.download(
        ["QQQ", "SHY"],
        start=signal_start.strftime("%Y-%m-%d"),
        end=fetch_end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned empty data for QQQ/SHY")

    opens = _flatten_yfinance_columns(raw, ["QQQ", "SHY"]).dropna(how="any")
    opens.index = pd.to_datetime(opens.index)
    week_ends = pd.date_range(signal_start, signal_end, freq="W-FRI")
    rows: list[dict[str, object]] = []
    for week_end in week_ends:
        future_dates = opens.index[opens.index > week_end]
        if future_dates.empty:
            raise RuntimeError(f"missing next-open execution date after {week_end.date()}")
        exec_date = future_dates[0]
        rows.append(
            {
                "week_end": week_end,
                "exec_date": exec_date,
                "exec_open_qqq": float(opens.at[exec_date, "open_qqq"]),
                "exec_open_shy": float(opens.at[exec_date, "open_shy"]),
            }
        )

    panel = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(cache, index=False)
    return panel


def _normalize_price_panel(price_panel: pd.DataFrame) -> pd.DataFrame:
    required = {"week_end", "exec_open_qqq", "exec_open_shy"}
    missing = required.difference(price_panel.columns)
    if missing:
        raise ValueError(f"price_panel missing required columns: {sorted(missing)}")

    panel = price_panel.copy()
    panel["week_end"] = pd.to_datetime(panel["week_end"])
    if "exec_date" in panel.columns:
        panel["exec_date"] = pd.to_datetime(panel["exec_date"])
        if not (panel["exec_date"] > panel["week_end"]).all():
            raise ValueError("execution dates must be strictly after week_end")
    else:
        panel["exec_date"] = pd.NaT

    for col in PRICE_COLUMNS:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    if panel[PRICE_COLUMNS].isna().any().any():
        raise ValueError("missing execution price in price_panel")
    if (panel[PRICE_COLUMNS] <= 0.0).any().any():
        raise ValueError("execution prices must be positive")

    return panel.sort_values("week_end", kind="mergesort").reset_index(drop=True)


def _normalize_weights(weights_df: pd.DataFrame) -> pd.DataFrame:
    required = {"week_end"}
    missing = required.difference(weights_df.columns)
    if missing:
        raise ValueError(f"weights_df missing required columns: {sorted(missing)}")

    weights = weights_df.copy()
    weights["week_end"] = pd.to_datetime(weights["week_end"])
    if "omega_qqq" not in weights.columns:
        if "omega_qqq_final" not in weights.columns:
            raise ValueError("weights_df missing omega_qqq or omega_qqq_final")
        weights["omega_qqq"] = weights["omega_qqq_final"]
    if "omega_shy" not in weights.columns:
        if "omega_shy_final" not in weights.columns:
            raise ValueError("weights_df missing omega_shy or omega_shy_final")
        weights["omega_shy"] = weights["omega_shy_final"]

    for col in ["omega_qqq", "omega_shy"]:
        weights[col] = pd.to_numeric(weights[col], errors="coerce")
    if weights[["omega_qqq", "omega_shy"]].isna().any().any():
        raise ValueError("weights_df contains missing weights")
    if (weights[["omega_qqq", "omega_shy"]] < 0.0).any().any():
        raise ValueError("weights_df contains negative weights")
    if not np.allclose(weights["omega_qqq"] + weights["omega_shy"], 1.0, atol=1e-10):
        raise ValueError("weights_df weights must sum to one")

    return weights[["week_end", "omega_qqq", "omega_shy"]].sort_values(
        "week_end",
        kind="mergesort",
    )


def run_backtest(
    weights_df: pd.DataFrame,
    price_panel: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Run the Phase 11 open-to-open weekly strategy backtest.

    Input:
        weights_df: Weekly final weights indexed by Friday week_end.
        price_panel: Execution-open panel aligned to the same week_end calendar.
        config: Backtest config carrying transaction cost settings.

    Output:
        DataFrame with one row per held period and cumulative NAV.

    Time/as-of semantics:
        For row t, the engine uses weights known at week_end[t], execution open
        at t, and execution open at t + 1 to compute the realized holding-period
        return. It does not reindex or search for future prices inside the loop.
    """

    weights = _normalize_weights(weights_df)
    panel = _normalize_price_panel(price_panel)
    aligned = weights.merge(panel, on="week_end", how="inner", validate="one_to_one")
    if len(aligned) < 2:
        raise ValueError("need at least two aligned execution rows for backtest")
    if len(aligned) != len(weights):
        raise ValueError("price_panel does not cover every weights_df week_end")

    aligned = aligned.sort_values("week_end", kind="mergesort").reset_index(drop=True)
    aligned["next_exec_open_qqq"] = aligned["exec_open_qqq"].shift(-1)
    aligned["next_exec_open_shy"] = aligned["exec_open_shy"].shift(-1)
    held = aligned.iloc[:-1].copy()

    held["qqq_return"] = held["next_exec_open_qqq"] / held["exec_open_qqq"] - 1.0
    held["shy_return"] = held["next_exec_open_shy"] / held["exec_open_shy"] - 1.0
    held["gross_portfolio_return"] = (
        held["omega_qqq"] * held["qqq_return"] + held["omega_shy"] * held["shy_return"]
    )

    prev_qqq = aligned["omega_qqq"].shift(1).iloc[:-1].copy()
    prev_qqq.iloc[0] = held["omega_qqq"].iloc[0]
    held["turnover"] = (held["omega_qqq"] - prev_qqq.to_numpy()).abs()
    held["transaction_cost"] = held["turnover"] * float(config.transaction_cost_bps) / 10_000.0
    held["net_portfolio_return"] = held["gross_portfolio_return"] - held["transaction_cost"]
    held["nav"] = (1.0 + held["net_portfolio_return"]).cumprod()

    output_cols = [
        "week_end",
        "exec_date",
        "omega_qqq",
        "omega_shy",
        "qqq_return",
        "shy_return",
        "gross_portfolio_return",
        "turnover",
        "transaction_cost",
        "net_portfolio_return",
        "nav",
    ]
    return held[output_cols].reset_index(drop=True)


def build_benchmark_nav(
    price_panel: pd.DataFrame,
    omega_qqq_fixed: float,
    omega_shy_fixed: float,
    cost_bps: float,
) -> pd.Series:
    """Build fixed-weight benchmark NAV on the Phase 11 execution calendar.

    Input:
        price_panel: Execution-open panel.
        omega_qqq_fixed, omega_shy_fixed: Long-only fixed benchmark weights.
        cost_bps: One-way transaction cost per unit turnover.

    Output:
        NAV series indexed by held-period week_end.

    Time/as-of semantics:
        Uses the same open-to-open period returns as the strategy. Fixed-weight
        benchmarks have zero turnover after initialization in this helper.
    """

    if not np.isclose(float(omega_qqq_fixed) + float(omega_shy_fixed), 1.0, atol=1e-12):
        raise ValueError("benchmark weights must sum to one")
    weights = pd.DataFrame(
        {
            "week_end": pd.to_datetime(price_panel["week_end"]),
            "omega_qqq_final": float(omega_qqq_fixed),
            "omega_shy_final": float(omega_shy_fixed),
        }
    )
    result = run_backtest(
        weights,
        price_panel,
        BacktestConfig(transaction_cost_bps=float(cost_bps), turnover_threshold=0.0),
    )
    nav = pd.Series(
        result["nav"].to_numpy(dtype=float),
        index=pd.to_datetime(result["week_end"]),
        name="nav",
    )
    nav.index = nav.index.strftime("%Y-%m-%d")
    return nav
