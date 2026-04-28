"""Unit tests for Phase 10 analysis functions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qqq_cycle.backtest.phase10_analysis import (
    HIGH_CONF_THRESHOLD,
    _max_drawdown,
    _ols_hac,
    _realized_at,
    check_signal_predictive_pass,
    cliffs_delta,
    compute_forward_metrics,
    compute_realized_metrics,
    load_strict_rows,
    run_regime_separation,
    run_signal_predictive,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PIPELINE_CSV = (
    Path(__file__).resolve().parents[1]
    / "outputs/pipeline/strict_real_pipeline_output.csv"
)


def _make_price_series(values: list[float], start: str = "2021-01-04") -> pd.Series:
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=dates, name="qqq_adj_close")


def _price_df(values: list[float], start: str = "2021-01-04") -> pd.DataFrame:
    s = _make_price_series(values, start)
    return s.to_frame()


# ---------------------------------------------------------------------------
# _max_drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_flat_series_is_zero() -> None:
    prices = pd.Series([100.0] * 5)
    assert _max_drawdown(prices) == pytest.approx(0.0)


def test_max_drawdown_monotone_decline() -> None:
    prices = pd.Series([100.0, 90.0, 80.0])
    # Peak is 100, trough is 80 → MDD = 0.20
    assert _max_drawdown(prices) == pytest.approx(0.20)


def test_max_drawdown_recovery() -> None:
    prices = pd.Series([100.0, 80.0, 120.0])
    # Peak was 100, worst trough 80 → MDD = 0.20
    assert _max_drawdown(prices) == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# _realized_at
# ---------------------------------------------------------------------------


def test_realized_at_r1w_correct() -> None:
    # 6 trading days: price doubles over 5 trading days
    prices = _make_price_series([100, 100, 100, 100, 100, 200])
    date = prices.index[-1]
    result = _realized_at(prices, date)
    assert result["r1w"] == pytest.approx(np.log(2.0), abs=1e-9)


def test_realized_at_returns_nan_insufficient_history() -> None:
    prices = _make_price_series([100.0])
    result = _realized_at(prices, prices.index[0])
    for k in ["r1w", "R4w", "sigma4w", "mdd8w"]:
        assert np.isnan(result[k])


def test_realized_at_sigma4w_positive() -> None:
    np.random.seed(42)
    rets = np.random.normal(0, 0.01, 30)
    prices_vals = 100 * np.cumprod(1 + rets)
    prices = _make_price_series(prices_vals.tolist())
    date = prices.index[-1]
    result = _realized_at(prices, date)
    assert result["sigma4w"] > 0


# ---------------------------------------------------------------------------
# compute_realized_metrics
# ---------------------------------------------------------------------------


def test_compute_realized_metrics_shape() -> None:
    np.random.seed(0)
    rets = np.random.normal(0.0005, 0.01, 60)
    prices_vals = 300 * np.cumprod(1 + rets)
    price_df = _price_df(prices_vals.tolist())
    week_ends = pd.Series(price_df.index[20::5])
    result = compute_realized_metrics(price_df, week_ends)
    assert len(result) == len(week_ends)
    assert set(result.columns) >= {"r1w", "R4w", "sigma4w", "mdd8w"}


# ---------------------------------------------------------------------------
# compute_forward_metrics
# ---------------------------------------------------------------------------


def test_compute_forward_metrics_1w_return() -> None:
    # Constant 1% weekly price increases
    weekly_prices = [100 * (1.01 ** i) for i in range(20)]
    # Make daily: repeat each price 5 times
    daily_prices = []
    for p in weekly_prices:
        daily_prices.extend([p] * 5)
    price_df = _price_df(daily_prices)
    week_end = price_df.index[4]  # End of first week
    result = compute_forward_metrics(price_df, pd.Series([week_end]), horizons=[1])
    ret = result.loc[week_end, "fwd_ret_1w"]
    assert abs(ret - np.log(1.01)) < 0.02  # Within 2% of log(1.01)


def test_compute_forward_metrics_nan_at_series_end() -> None:
    prices_vals = [100.0] * 30
    price_df = _price_df(prices_vals)
    last_date = price_df.index[-1]
    result = compute_forward_metrics(price_df, pd.Series([last_date]), horizons=[4])
    assert np.isnan(result.loc[last_date, "fwd_ret_4w"])


# ---------------------------------------------------------------------------
# cliffs_delta
# ---------------------------------------------------------------------------


def test_cliffs_delta_all_x_greater_than_y() -> None:
    x = np.array([10.0, 11.0, 12.0])
    y = np.array([1.0, 2.0, 3.0])
    assert cliffs_delta(x, y) == pytest.approx(1.0)


def test_cliffs_delta_all_x_less_than_y() -> None:
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 11.0, 12.0])
    assert cliffs_delta(x, y) == pytest.approx(-1.0)


def test_cliffs_delta_identical_arrays() -> None:
    x = np.array([5.0, 5.0, 5.0])
    assert cliffs_delta(x, x) == pytest.approx(0.0)


def test_cliffs_delta_nan_inputs_returns_nan() -> None:
    assert np.isnan(cliffs_delta(np.array([np.nan]), np.array([1.0])))


# ---------------------------------------------------------------------------
# _ols_hac
# ---------------------------------------------------------------------------


def test_ols_hac_recovers_true_beta() -> None:
    np.random.seed(7)
    n = 100
    x = np.random.randn(n)
    y = 2.0 + 1.5 * x + np.random.randn(n) * 0.5
    beta, tstat, pval = _ols_hac(y, x, lag=4)
    assert abs(beta - 1.5) < 0.3
    assert pval < 0.01


def test_ols_hac_zero_signal_insignificant() -> None:
    np.random.seed(99)
    n = 80
    x = np.random.randn(n)
    y = np.random.randn(n)
    _, tstat, pval = _ols_hac(y, x, lag=4)
    assert pval > 0.05


def test_ols_hac_positive_beta_for_correlated_series() -> None:
    np.random.seed(42)
    x = np.linspace(0, 1, 50) + np.random.randn(50) * 0.05
    y = 3.0 * x + np.random.randn(50) * 0.1
    beta, _, _ = _ols_hac(y, x, lag=3)
    assert beta > 0


# ---------------------------------------------------------------------------
# run_regime_separation with synthetic data
# ---------------------------------------------------------------------------


def _make_synthetic_analysis_df(n_per_state: int = 30) -> pd.DataFrame:
    """Create synthetic merged DataFrame with clear state separation."""
    np.random.seed(123)
    records = []
    # States 0-1: low vol/MDD; states 3-4: high vol/MDD
    state_vol = {0: 0.05, 1: 0.08, 2: 0.15, 3: 0.28, 4: 0.35}
    state_mdd = {0: 0.02, 1: 0.04, 2: 0.10, 3: 0.20, 4: 0.30}
    for state, vol in state_vol.items():
        for _ in range(n_per_state):
            records.append({
                "k_hat_t": state,
                "h_t": np.random.uniform(0, 1),
                "rho_t": np.random.uniform(0, 1),
                "s_t": np.random.uniform(0, 1),
                "I_t": np.random.randn(),
                "r1w": np.random.normal(0, vol * 0.3),
                "R4w": np.random.normal(0, vol * 0.6),
                "sigma4w": np.random.normal(vol, vol * 0.1),
                "mdd8w": np.random.normal(state_mdd[state], state_mdd[state] * 0.15),
                "max_p_t": np.random.uniform(0.4, 0.9),
            })
    return pd.DataFrame(records)


def test_regime_separation_passes_with_clear_separation() -> None:
    df = _make_synthetic_analysis_df(n_per_state=40)
    results = run_regime_separation(df)
    assert results["passed"] is True


def test_regime_separation_summary_has_all_states() -> None:
    df = _make_synthetic_analysis_df(n_per_state=20)
    results = run_regime_separation(df)
    summary_df = pd.DataFrame(results["summary_records"])
    states_in_summary = summary_df["state"].unique()
    assert set(states_in_summary) == {0, 1, 2, 3, 4}


def test_regime_separation_tests_has_kw_for_each_metric() -> None:
    df = _make_synthetic_analysis_df(n_per_state=20)
    results = run_regime_separation(df)
    for metric in ["r1w", "R4w", "sigma4w", "mdd8w"]:
        assert metric in results["tests"]
        assert "kw_pvalue" in results["tests"][metric]


def test_regime_separation_fails_with_no_separation() -> None:
    np.random.seed(0)
    n = 150
    df = pd.DataFrame({
        "k_hat_t": np.random.randint(0, 5, n),
        "sigma4w": np.random.normal(0.15, 0.001, n),
        "mdd8w": np.random.normal(0.10, 0.001, n),
        "r1w": np.random.randn(n) * 0.01,
        "R4w": np.random.randn(n) * 0.02,
    })
    results = run_regime_separation(df)
    assert results["passed"] is False


# ---------------------------------------------------------------------------
# run_signal_predictive with synthetic data
# ---------------------------------------------------------------------------


def _make_signal_predictive_df(n: int = 80) -> pd.DataFrame:
    """Synthetic data where rho_t predicts fwd_vol and fwd_mdd positively."""
    np.random.seed(55)
    rho_t = np.random.uniform(0.1, 0.9, n)
    h_t = rho_t + np.random.randn(n) * 0.1
    noise = np.random.randn(n)
    df = pd.DataFrame({
        "h_t": np.clip(h_t, 0, 1),
        "rho_t": rho_t,
        "s_t": np.random.uniform(0, 1, n),
        "I_t": np.random.randn(n),
        "fwd_ret_1w": -0.5 * rho_t + noise * 0.05,
        "fwd_ret_4w": -0.8 * rho_t + noise * 0.08,
        "fwd_ret_8w": -1.0 * rho_t + noise * 0.10,
        "fwd_vol_1w": 0.3 * rho_t + 0.05 + noise * 0.01,
        "fwd_vol_4w": 0.4 * rho_t + 0.05 + noise * 0.015,
        "fwd_vol_8w": 0.5 * rho_t + 0.05 + noise * 0.02,
        "fwd_mdd_1w": 0.2 * rho_t + 0.01 + noise * 0.01,
        "fwd_mdd_4w": 0.3 * rho_t + 0.01 + noise * 0.015,
        "fwd_mdd_8w": 0.4 * rho_t + 0.01 + noise * 0.02,
        "max_p_t": np.random.uniform(0.4, 0.9, n),
        "k_hat_t": np.random.randint(0, 5, n),
    })
    return df


def test_signal_predictive_returns_expected_shape() -> None:
    df = _make_signal_predictive_df()
    high_conf = df[df["max_p_t"] >= HIGH_CONF_THRESHOLD]
    result = run_signal_predictive(df, high_conf)
    # 4 signals × 3 horizons × 3 targets × 2 subsamples = 72
    assert len(result) == 72
    assert "spearman_rho" in result.columns
    assert "hac_pvalue" in result.columns
    assert "tercile_spread" in result.columns


def test_signal_predictive_pass_with_strong_signal() -> None:
    df = _make_signal_predictive_df(n=120)
    high_conf = df[df["max_p_t"] >= HIGH_CONF_THRESHOLD]
    result = run_signal_predictive(df, high_conf)
    passed = check_signal_predictive_pass(result)
    assert passed is True


def test_signal_predictive_fail_with_noise_only() -> None:
    np.random.seed(77)
    n = 80
    df = pd.DataFrame({
        "h_t": np.random.uniform(0, 1, n),
        "rho_t": np.random.uniform(0, 1, n),
        "s_t": np.random.uniform(0, 1, n),
        "I_t": np.random.randn(n),
        "fwd_ret_1w": np.random.randn(n) * 0.01,
        "fwd_ret_4w": np.random.randn(n) * 0.02,
        "fwd_ret_8w": np.random.randn(n) * 0.03,
        "fwd_vol_1w": np.random.uniform(0.1, 0.3, n),
        "fwd_vol_4w": np.random.uniform(0.1, 0.3, n),
        "fwd_vol_8w": np.random.uniform(0.1, 0.3, n),
        "fwd_mdd_1w": np.random.uniform(0.01, 0.15, n),
        "fwd_mdd_4w": np.random.uniform(0.01, 0.15, n),
        "fwd_mdd_8w": np.random.uniform(0.01, 0.15, n),
        "max_p_t": np.random.uniform(0.4, 0.9, n),
    })
    high_conf = df[df["max_p_t"] >= HIGH_CONF_THRESHOLD]
    result = run_signal_predictive(df, high_conf)
    passed = check_signal_predictive_pass(result)
    assert passed is False


# ---------------------------------------------------------------------------
# Integration smoke: load_strict_rows on real pipeline output
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PIPELINE_CSV.exists(),
    reason="Pipeline output not found; run scripts/run_pipeline.py first",
)
def test_load_strict_rows_returns_nonempty() -> None:
    df = load_strict_rows(PIPELINE_CSV)
    assert len(df) > 0
    assert "h_t" in df.columns
    assert "rho_t" in df.columns
    assert df["mode"].unique().tolist() == ["strict"]


@pytest.mark.skipif(
    not PIPELINE_CSV.exists(),
    reason="Pipeline output not found",
)
def test_load_strict_rows_all_contracts_satisfied() -> None:
    df = load_strict_rows(PIPELINE_CSV)
    assert df["strict_contracts_satisfied"].all()
