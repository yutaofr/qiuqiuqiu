import pandas as pd
import numpy as np

from qqq_cycle.core.risk_layer import (
    PRODUCTION_RISK_ENABLED,
    compute_ewcorr_78w,
    compute_risk_score,
    blended_state_weight,
)


def test_risk_layer_production_rho_is_enabled() -> None:
    assert PRODUCTION_RISK_ENABLED is True


def test_blended_state_weight_uses_natural_weight_below_drift_band() -> None:
    p_t = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    omega = np.array([1.0, 0.7, 0.3, 0.6, 0.9])

    out = blended_state_weight(p_t, omega, delta_abs_raw=1.0)

    assert out == 1.0


def test_blended_state_weight_uses_neutral_above_drift_band() -> None:
    p_t = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    omega = np.array([1.0, 0.7, 0.3, 0.6, 0.9])

    out = blended_state_weight(p_t, omega, delta_abs_raw=-2.0)

    assert out == 0.6


def test_blended_state_weight_is_linear_inside_drift_band() -> None:
    p_t = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    omega = np.array([1.0, 0.7, 0.3, 0.6, 0.9])

    out = blended_state_weight(p_t, omega, delta_abs_raw=1.5)

    assert np.isclose(out, 0.8)


def test_compute_risk_score_matches_maximum_formula() -> None:
    out = compute_risk_score(omega_t=1.0, s_t=1.0, h_t_lead=1.0, lambda_rho=0.75)

    assert out.rho_t == 1.0
    assert out.m_t == 1.0
    assert out.n_t == 1.0


def test_compute_risk_score_is_zero_without_state_or_micro_signal() -> None:
    out = compute_risk_score(omega_t=0.6, s_t=0.0, h_t_lead=0.5, lambda_rho=0.75)

    assert out.rho_t == 0.0
    assert out.m_t == 0.0
    assert out.n_t == 0.0


def test_compute_risk_score_stays_in_unit_interval_for_random_inputs() -> None:
    rng = np.random.default_rng(20260427)
    for omega_t, s_t, h_t_lead in rng.random((1000, 3)):
        out = compute_risk_score(omega_t=omega_t, s_t=s_t, h_t_lead=0.5 + 0.5 * h_t_lead)
        assert 0.0 <= out.rho_t <= 1.0


def test_compute_risk_score_monotonic_in_state_and_micro_inputs() -> None:
    low_s = compute_risk_score(omega_t=0.8, s_t=0.2, h_t_lead=0.7).rho_t
    high_s = compute_risk_score(omega_t=0.8, s_t=0.4, h_t_lead=0.7).rho_t
    low_h = compute_risk_score(omega_t=0.8, s_t=0.2, h_t_lead=0.6).rho_t
    high_h = compute_risk_score(omega_t=0.8, s_t=0.2, h_t_lead=0.8).rho_t

    assert high_s > low_s
    assert high_h > low_h


def test_compute_ewcorr_78w_returns_interpretability_only_series() -> None:
    idx = pd.date_range("2020-01-03", periods=90, freq="W-FRI")
    s = pd.Series(np.linspace(0.0, 1.0, len(idx)), index=idx)
    h = pd.Series(np.linspace(0.5, 1.0, len(idx)), index=idx)

    eta = compute_ewcorr_78w(s, h)

    assert eta.index.equals(idx)
    assert eta.iloc[-1] > 0.99
