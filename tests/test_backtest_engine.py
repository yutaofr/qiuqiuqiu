"""Tests for Phase 11 no-lookahead backtest engine."""

from __future__ import annotations

import pandas as pd
import pytest

from qqq_cycle.backtest.engine import build_benchmark_nav, run_backtest
from qqq_cycle.portfolio.construction import BacktestConfig


def _weights(values: list[float]) -> pd.DataFrame:
    week_ends = pd.date_range("2024-01-05", periods=len(values), freq="W-FRI")
    return pd.DataFrame(
        {
            "week_end": week_ends,
            "omega_qqq_final": values,
            "omega_shy_final": [1.0 - v for v in values],
        }
    )


def _price_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week_end": pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"]),
            "exec_date": pd.to_datetime(["2024-01-08", "2024-01-16", "2024-01-22"]),
            "exec_open_qqq": [100.0, 110.0, 121.0],
            "exec_open_shy": [100.0, 100.0, 100.0],
        }
    )


def test_next_open_execution_timing() -> None:
    result = run_backtest(_weights([1.0, 1.0, 1.0]), _price_panel(), BacktestConfig())

    assert result.loc[0, "qqq_return"] == pytest.approx(0.10)
    assert result.loc[0, "gross_portfolio_return"] == pytest.approx(0.10)


def test_transaction_cost_applied_on_turnover() -> None:
    result = run_backtest(_weights([0.50, 0.70, 0.70]), _price_panel(), BacktestConfig())

    assert result.loc[1, "turnover"] == pytest.approx(0.20)
    assert result.loc[1, "transaction_cost"] == pytest.approx(0.20 * 0.0005)


def test_nav_path_reproducible() -> None:
    weights = _weights([0.60, 0.40, 0.80])
    prices = _price_panel()

    first = run_backtest(weights, prices, BacktestConfig())
    second = run_backtest(weights, prices, BacktestConfig())

    pd.testing.assert_series_equal(first["nav"], second["nav"])


def test_no_lookahead_in_price_alignment() -> None:
    result = run_backtest(_weights([1.0, 1.0, 1.0]), _price_panel(), BacktestConfig())

    week_end = pd.to_datetime(result["week_end"])
    exec_date = pd.to_datetime(result["exec_date"])
    assert (exec_date > week_end).all()


def test_missing_execution_price_raises() -> None:
    prices = _price_panel()
    prices.loc[1, "exec_open_qqq"] = float("nan")

    with pytest.raises(ValueError, match="missing execution price"):
        run_backtest(_weights([1.0, 1.0, 1.0]), prices, BacktestConfig())


def test_benchmark_nav_uses_same_open_to_open_calendar() -> None:
    nav = build_benchmark_nav(_price_panel(), 1.0, 0.0, cost_bps=5.0)

    assert list(nav.index.astype(str)) == ["2024-01-05", "2024-01-12"]
    assert nav.iloc[0] == pytest.approx(1.10)
    assert nav.iloc[1] == pytest.approx(1.21)
