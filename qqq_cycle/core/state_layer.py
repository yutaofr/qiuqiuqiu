"""Macro state-layer factors L_t, T_t, P_t, E_t, and Theta_t."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qqq_cycle.core.dual_memory import dual_memory, exo_dual_memory


def realized_vol_20w(price: pd.Series) -> pd.Series:
    """Annualized 20-week realized volatility from PIT weekly closes."""

    log_ret = np.log(price.astype(float)).diff()
    return log_ret.rolling(20, min_periods=20).std() * np.sqrt(52.0)


def compute_liquidity_factor(
    dfii10: pd.Series,
    dgs2: pd.Series,
    hyoas: pd.Series,
    nfci: pd.Series,
    *,
    eps: float = 1e-12,
) -> pd.Series:
    """Compute L_t from standardized liquidity inputs known as of week t."""

    dgs2_delta4 = dgs2.astype(float).diff(4)
    return 0.25 * (
        -dual_memory(dfii10, 104, 260, eps)
        - dual_memory(dgs2_delta4, 104, 260, eps)
        - dual_memory(hyoas, 104, 260, eps)
        - dual_memory(nfci, 104, 260, eps)
    )


def compute_temperature_factor(qqq: pd.Series, *, eps: float = 1e-12) -> pd.Series:
    """Compute T_t from QQQ deviations versus 52w and 156w moving averages."""

    q = qqq.astype(float)
    u1 = q / q.rolling(52, min_periods=52).mean() - 1.0
    u2 = q / q.rolling(156, min_periods=156).mean() - 1.0
    return 0.5 * dual_memory(u1, 104, 260, eps) + 0.5 * dual_memory(u2, 104, 260, eps)


def compute_risk_preference_factor(
    vix: pd.Series, qqq: pd.Series, *, eps: float = 1e-12
) -> pd.Series:
    """Compute P_t from VIX, 20w realized volatility, and QQQ/MA40."""

    q = qqq.astype(float)
    rv = realized_vol_20w(q)
    ma40_dev = q / q.rolling(40, min_periods=40).mean() - 1.0
    return (1.0 / 3.0) * (
        -dual_memory(vix, 104, 260, eps)
        - dual_memory(rv, 104, 260, eps)
        + dual_memory(ma40_dev, 104, 260, eps)
    )


def compute_exogenous_factor(
    ai_gpr: pd.Series, usepuindxd: pd.Series, *, eps: float = 1e-12
) -> pd.Series:
    """Compute E_t from exogenous dual-memory normalized AI-GPR and EPU."""

    return 0.5 * exo_dual_memory(ai_gpr, eps) + 0.5 * exo_dual_memory(usepuindxd, eps)


def compute_theta(L_t: pd.Series, T_t: pd.Series, P_t: pd.Series) -> pd.DataFrame:
    """Compute state coordinates Theta_t = [H_t, I_t]."""

    h = 0.40 * L_t + 0.35 * T_t + 0.25 * P_t
    i = 0.50 * L_t.diff(4) + 0.30 * T_t.diff(4) + 0.20 * P_t.diff(4)
    return pd.DataFrame({"H": h, "I": i}, index=L_t.index)


def compute_state_layer(inputs: pd.DataFrame, *, eps: float = 1e-12) -> pd.DataFrame:
    """Compute first-slice macro factors from aligned weekly input columns.

    Required columns: DFII10, DGS2, BAMLH0A0HYM2, NFCI, VIXCLS, AI_GPR,
    USEPUINDXD, QQQ.
    """

    hyoas = inputs["BAMLH0A0HYM2"]
    l_t = compute_liquidity_factor(inputs["DFII10"], inputs["DGS2"], hyoas, inputs["NFCI"], eps=eps)
    t_t = compute_temperature_factor(inputs["QQQ"], eps=eps)
    p_t = compute_risk_preference_factor(inputs["VIXCLS"], inputs["QQQ"], eps=eps)
    e_t = compute_exogenous_factor(inputs["AI_GPR"], inputs["USEPUINDXD"], eps=eps)
    theta = compute_theta(l_t, t_t, p_t)
    return pd.DataFrame(
        {
            "L": l_t,
            "T": t_t,
            "P": p_t,
            "E": e_t,
            "H": theta["H"],
            "I": theta["I"],
        },
        index=inputs.index,
    )
