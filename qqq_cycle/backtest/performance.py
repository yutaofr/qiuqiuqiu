"""Phase 11 performance metrics and benchmark comparison tables."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


WEEKS_PER_YEAR = 52.0
PERFORMANCE_METRICS = [
    "annualized_return",
    "annualized_vol",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "annualized_turnover",
    "cost_drag",
    "hit_ratio",
    "worst_week",
    "worst_4w_return",
]


def _clean_returns(returns: pd.Series) -> pd.Series:
    out = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    if out.empty:
        raise ValueError("returns series is empty after dropping NaN")
    return out.astype(float)


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    nav = (1.0 + returns).cumprod()
    peaks = nav.cummax()
    drawdowns = nav / peaks - 1.0
    return float(-drawdowns.min())


def compute_performance_metrics(
    returns: pd.Series,
    rf: float = 0.0,
    *,
    turnover: pd.Series | None = None,
    transaction_cost: pd.Series | None = None,
) -> dict[str, float]:
    """Compute the frozen Phase 11 performance metric set.

    Input:
        returns: Net weekly returns on the Phase 11 execution calendar.
        rf: Annual risk-free rate, pre-registered as zero.
        turnover: Optional weekly turnover series for annualized turnover.
        transaction_cost: Optional weekly transaction-cost return drag.

    Output:
        Dictionary containing all PERFORMANCE_METRICS keys.

    Time/as-of semantics:
        This is an ex-post reporting function over already-computed backtest
        returns. It does not alter signals, weights, or execution prices.
    """

    r = _clean_returns(returns)
    n = len(r)
    years = n / WEEKS_PER_YEAR
    total_return = float((1.0 + r).prod() - 1.0)
    annualized_return = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else np.nan
    annualized_vol = float(r.std(ddof=1) * math.sqrt(WEEKS_PER_YEAR)) if n > 1 else 0.0
    weekly_rf = float(rf) / WEEKS_PER_YEAR
    excess = r - weekly_rf
    sharpe = float(excess.mean() * WEEKS_PER_YEAR / annualized_vol) if annualized_vol > 0 else np.nan

    downside = np.minimum(excess.to_numpy(dtype=float), 0.0)
    downside_dev = float(np.sqrt(np.mean(downside**2)) * math.sqrt(WEEKS_PER_YEAR))
    sortino = float(excess.mean() * WEEKS_PER_YEAR / downside_dev) if downside_dev > 0 else np.nan

    max_drawdown = _max_drawdown_from_returns(r)
    calmar = float(annualized_return / max_drawdown) if max_drawdown > 0 else np.nan

    if turnover is None:
        annualized_turnover = 0.0
    else:
        annualized_turnover = float(pd.to_numeric(turnover, errors="coerce").fillna(0.0).mean() * WEEKS_PER_YEAR)

    if transaction_cost is None:
        cost_drag = 0.0
    else:
        costs = pd.to_numeric(transaction_cost, errors="coerce").fillna(0.0)
        cost_drag = float(costs.sum() / years) if years > 0 else np.nan

    rolling_4w = (1.0 + r).rolling(4).apply(np.prod, raw=True) - 1.0
    worst_4w = float(rolling_4w.min()) if rolling_4w.notna().any() else np.nan

    return {
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "annualized_turnover": annualized_turnover,
        "cost_drag": cost_drag,
        "hit_ratio": float((r > 0.0).mean()),
        "worst_week": float(r.min()),
        "worst_4w_return": worst_4w,
    }


def build_comparison_table(
    strategy_returns: pd.Series,
    qqq_returns: pd.Series,
    shy_returns: pd.Series,
    static_6040_returns: pd.Series,
) -> pd.DataFrame:
    """Build the Phase 11 strategy/benchmark metric table.

    Input:
        Four weekly return series on the same execution calendar.

    Output:
        DataFrame indexed by metric names with columns strategy, qqq_buyhold,
        shy_buyhold, static_6040.

    Time/as-of semantics:
        Reporting-only comparison over realized backtest returns.
    """

    metrics = {
        "strategy": compute_performance_metrics(strategy_returns),
        "qqq_buyhold": compute_performance_metrics(qqq_returns),
        "shy_buyhold": compute_performance_metrics(shy_returns),
        "static_6040": compute_performance_metrics(static_6040_returns),
    }
    return pd.DataFrame(metrics).reindex(PERFORMANCE_METRICS)
