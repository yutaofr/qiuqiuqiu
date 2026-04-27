"""Production risk-layer utilities for rho_t and interpretability eta_t."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PRODUCTION_RISK_ENABLED = True


@dataclass(frozen=True)
class RiskScore:
    """Production rho_t components.

    Inputs are weekly values knowable at the decision timestamp:
    `omega_t` from state/drift blending, `s_t` from stress, and `h_t_lead` from
    the micro-layer IIR envelope. Output is clipped to the model's [0, 1] range.
    """

    m_t: float
    n_t: float
    rho_t: float


def blended_state_weight(
    p_t: np.ndarray,
    omega_state: np.ndarray,
    delta_abs_raw: float,
    theta_lo: float = 1.2,
    theta_hi: float = 1.8,
    neutral: float = 0.6,
) -> float:
    """Return drift-degraded state condition weight.

    Input:
        p_t: Five-state probability vector.
        omega_state: Five fixed semantic state weights.
        delta_abs_raw: Physical-space drift degree.
    Output:
        Scalar state weight after linear blending to neutral weight.
    Scope:
        Production state-condition component for model §10.1.
    """

    probs = np.asarray(p_t, dtype=float)
    weights = np.asarray(omega_state, dtype=float)
    if probs.shape != (5,) or weights.shape != (5,):
        raise ValueError("p_t and omega_state must both have shape (5,)")
    if theta_hi <= theta_lo:
        raise ValueError("theta_hi must be greater than theta_lo")
    if not np.isclose(probs.sum(), 1.0, atol=1e-8):
        raise ValueError("p_t must sum to 1")
    omega_nat = float(np.dot(probs, weights))
    alpha = float(np.clip((abs(delta_abs_raw) - theta_lo) / (theta_hi - theta_lo), 0.0, 1.0))
    return (1.0 - alpha) * omega_nat + alpha * neutral


def compute_risk_score(
    *,
    omega_t: float,
    s_t: float,
    h_t_lead: float,
    lambda_rho: float = 0.75,
) -> RiskScore:
    """Compute production §10.2-§10.3 rho_t.

    Args:
        omega_t: Drift-degraded state condition weight.
        s_t: Direction-neutral stress score in [0, 1].
        h_t_lead: Weekly micro fragility lead score, where 0.5 is no signal.
        lambda_rho: Fixed micro amplification coefficient.

    Returns:
        RiskScore with m_t, n_t, and rho_t in [0, 1].
    """

    if not 0.0 <= lambda_rho <= 1.0:
        raise ValueError("lambda_rho must be in [0, 1]")
    m_t = float(np.clip(float(omega_t) * float(s_t), 0.0, 1.0))
    n_t = float(np.clip(2.0 * max(float(h_t_lead) - 0.5, 0.0), 0.0, 1.0))
    rho_t = float(1.0 - (1.0 - m_t) * (1.0 - lambda_rho * n_t))
    return RiskScore(m_t=m_t, n_t=n_t, rho_t=float(np.clip(rho_t, 0.0, 1.0)))


def compute_ewcorr_78w(
    s_series: pd.Series,
    h_lead_series: pd.Series,
    *,
    half_life: int = 78,
    eps: float = 1e-12,
) -> pd.Series:
    """Return EWCorr_78w(s_t, h_t_lead) for interpretability only.

    The recurrence is point-in-time: each row updates EW means, covariance, and
    variances using only observations through that row.
    """

    if half_life < 1:
        raise ValueError("half_life must be >= 1")
    s = pd.Series(s_series, dtype=float)
    h = pd.Series(h_lead_series, dtype=float).reindex(s.index)
    rho = 2 ** (-1 / half_life)
    mu_s = np.nan
    mu_h = np.nan
    var_s = 0.0
    var_h = 0.0
    cov = 0.0
    out = pd.Series(np.nan, index=s.index, dtype=float)
    for idx, s_value, h_value in zip(s.index, s.to_numpy(dtype=float), h.to_numpy(dtype=float)):
        if not np.isfinite(s_value) or not np.isfinite(h_value):
            continue
        if not np.isfinite(mu_s):
            mu_s = float(s_value)
            mu_h = float(h_value)
            out.loc[idx] = np.nan
            continue
        ds = float(s_value) - mu_s
        dh = float(h_value) - mu_h
        cov = rho * cov + (1.0 - rho) * ds * dh
        var_s = rho * var_s + (1.0 - rho) * ds * ds
        var_h = rho * var_h + (1.0 - rho) * dh * dh
        mu_s = rho * mu_s + (1.0 - rho) * float(s_value)
        mu_h = rho * mu_h + (1.0 - rho) * float(h_value)
        out.loc[idx] = cov / (np.sqrt(var_s * var_h) + eps)
    return out.clip(lower=-1.0, upper=1.0)
