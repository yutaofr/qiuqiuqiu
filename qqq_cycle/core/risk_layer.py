"""Risk-layer interface utilities without production rho_t replay."""

from __future__ import annotations

import numpy as np

INTERFACE_ONLY = "INTERFACE-ONLY / NOT PRODUCTION RISK"
PRODUCTION_RISK_ENABLED = False


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
        This is only the interface utility from model §10.1. It does not compute
        production `rho_t` or combine micro-layer outputs.
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
