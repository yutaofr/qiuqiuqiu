"""State stress layer for direction-neutral motion strength."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qqq_cycle.core.covariance import CovarianceState2D, RobustEWCov2D, regularize_cov_2d
from qqq_cycle.core.dual_memory import z_rob


def logistic(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable standard logistic Lambda."""

    arr = np.asarray(x, dtype=float)
    out = np.empty_like(arr, dtype=float)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    exp_x = np.exp(arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    if np.isscalar(x):
        return float(out)
    return out


@dataclass
class StressResult:
    """Stress-layer output table and diagnostic states."""

    frame: pd.DataFrame
    velocity_cov_state: CovarianceState2D
    acceleration_cov_state: CovarianceState2D


def rolling_noise_floor(delta_v: np.ndarray, window: int = 520, quantile: float = 0.10) -> np.ndarray:
    """Return exact expanding/rolling diagonal quantile noise floors."""

    delta = np.asarray(delta_v, dtype=float)
    out = np.zeros((len(delta), 2, 2), dtype=float)
    sq = delta**2
    for i in range(len(delta)):
        start = max(0, i - window + 1)
        vals = sq[start : i + 1]
        q = np.nanquantile(vals, quantile, axis=0)
        out[i] = np.diag(np.maximum(q, 1e-12))
    return out


def compute_stress_layer(
    theta: pd.DataFrame,
    e_t: pd.Series,
    *,
    warmup_history: np.ndarray | None = None,
) -> StressResult:
    """Compute velocity, acceleration, raw stress, pre-sigmoid score, and s_t.

    Input:
        theta: DataFrame with H and I columns, indexed by decision week.
        e_t: Exogenous factor series aligned to theta.
    Output:
        StressResult with columns v_H, v_I, d, a, g_raw, g_stress, s.
    Time semantics:
        Updates scan forward once. NaN Theta rows advance missing-step counters
        in the covariance states and do not impute observations.
    """

    values = theta[["H", "I"]].to_numpy(dtype=float)
    if warmup_history is None:
        finite = values[np.isfinite(values).all(axis=1)]
        if len(finite) < 2:
            finite = np.array([[0.0, 0.0], [1e-6, 0.0]])
        warmup_history = finite[: max(2, min(len(finite), 20))]
    cov_v = RobustEWCov2D()
    cov_a = RobustEWCov2D()
    state_v = cov_v.initialize_from_history(warmup_history)
    state_a = cov_a.initialize_from_history(warmup_history)
    rho_v = 2 ** (-1 / 4)
    pending_velocity_missing = 0
    prev_theta: np.ndarray | None = None
    prev_v = np.zeros(2, dtype=float)
    rows: list[dict[str, float]] = []
    delta_v_hist: list[np.ndarray] = []

    for x in values:
        if np.any(~np.isfinite(x)) or prev_theta is None:
            if prev_theta is not None:
                pending_velocity_missing += 1
                state_v = cov_v.update(state_v, np.array([np.nan, np.nan]))
                state_a = cov_a.update(state_a, np.array([np.nan, np.nan]))
            if np.isfinite(x).all() and prev_theta is None:
                prev_theta = x
            rows.append({"v_H": np.nan, "v_I": np.nan, "d": np.nan, "a": np.nan, "g_raw": np.nan})
            continue

        k = pending_velocity_missing + 1
        effective_rho_v = rho_v**k
        delta_theta = x - prev_theta
        v = effective_rho_v * prev_v + (1 - effective_rho_v) * delta_theta
        delta_a = v - prev_v
        delta_v_hist.append(delta_a)

        state_v = cov_v.update(state_v, v)
        state_a = cov_a.update(state_a, delta_a)
        hist = np.asarray(delta_v_hist[-520:], dtype=float)
        q = np.nanquantile(hist**2, 0.10, axis=0)
        noise = np.diag(np.maximum(q, 1e-12))
        state_a.cov_raw = state_a.cov_raw + noise
        state_a.cov_reg, state_a.eigvals, state_a.eigvecs = regularize_cov_2d(
            state_a.cov_raw,
            cov_a.eps_abs,
            cov_a.eps_rel,
        )

        d = float(np.sqrt(v.T @ np.linalg.inv(state_v.cov_reg) @ v))
        a = float(np.sqrt(delta_a.T @ np.linalg.inv(state_a.cov_reg) @ delta_a))
        rows.append(
            {
                "v_H": v[0],
                "v_I": v[1],
                "d": d,
                "a": a,
                "g_raw": 0.5 * d + 0.5 * a,
            }
        )
        prev_theta = x
        prev_v = v
        pending_velocity_missing = 0

    frame = pd.DataFrame(rows, index=theta.index)
    g_z = z_rob(frame["g_raw"], window=156, eps=1e-12)
    e_z = z_rob(e_t, window=156, eps=1e-12)
    frame["g_stress"] = 0.5 * g_z + 0.5 * e_z
    frame["s"] = pd.Series(logistic(frame["g_stress"].to_numpy()), index=theta.index)
    frame.loc[frame["g_stress"].isna(), "s"] = np.nan
    return StressResult(frame=frame, velocity_cov_state=state_v, acceleration_cov_state=state_a)
