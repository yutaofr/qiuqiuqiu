"""Robust 2D EW covariance with spectral flooring and health diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
COND_WARN_RATIO = 0.9


@dataclass
class CovarianceState2D:
    """2D covariance recursion state.

    pending_missing_steps counts elapsed NaN weeks whose decay will be applied
    to the next valid observation with weight complement `1 - rho**k`.
    """

    mean: np.ndarray
    cov_raw: np.ndarray
    cov_reg: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    warmup_count: int
    pending_missing_steps: int = 0
    last_diagnostics: dict[str, Any] | None = field(default=None)
    state_ok: bool = True


def _raw_eigen_diagnostics(
    cov_raw: np.ndarray, eps_abs: float, eps_rel: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    eigvals_raw, eigvecs = np.linalg.eigh((cov_raw + cov_raw.T) / 2.0)
    order = np.argsort(eigvals_raw)[::-1]
    eigvals_raw = eigvals_raw[order]
    eigvecs = eigvecs[:, order]
    lam1, lam2 = float(eigvals_raw[0]), float(eigvals_raw[1])
    if lam1 < -eps_abs or lam2 < -eps_abs:
        raise np.linalg.LinAlgError("negative covariance eigenvalue")
    lam1 = max(lam1, eps_abs)
    lam2_reg = max(lam2, eps_rel * lam1, eps_abs)
    diagnostics = {
        "eigval_1": lam1,
        "eigval_2_raw": lam2,
        "eigval_2_reg": lam2_reg,
        "condition_number_raw": lam1 / max(lam2, 1e-15),
        "condition_number_reg": lam1 / lam2_reg,
        "eigval_2_was_floored": bool(lam2 < eps_rel * lam1 or lam2 < eps_abs),
    }
    return np.array([lam1, lam2_reg], dtype=float), eigvecs, diagnostics


def regularize_cov_2d(
    cov_raw: np.ndarray,
    eps_abs: float = 1e-8,
    eps_rel: float = 1e-4,
    *,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[
    np.ndarray, np.ndarray, np.ndarray, dict[str, Any]
]:
    """Apply selective secondary-eigenvalue flooring to a 2D covariance."""

    cov = np.asarray(cov_raw, dtype=float)
    if cov.shape != (2, 2):
        raise ValueError("cov_raw must have shape (2, 2)")
    eigvals_star, eigvecs, diagnostics = _raw_eigen_diagnostics(cov, eps_abs, eps_rel)
    cov_reg = eigvecs @ np.diag(eigvals_star) @ eigvecs.T
    cov_reg = (cov_reg + cov_reg.T) / 2.0
    if return_diagnostics:
        return cov_reg, eigvals_star, eigvecs, diagnostics
    return cov_reg, eigvals_star, eigvecs


def symmetric_sqrt_2d(cov: np.ndarray) -> np.ndarray:
    """Return symmetric spectral square root of a 2D positive covariance."""

    cov_reg, eigvals, eigvecs = regularize_cov_2d(cov)
    del cov_reg
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def symmetric_inv_sqrt_2d(cov: np.ndarray) -> np.ndarray:
    """Return symmetric spectral inverse square root of a 2D covariance."""

    cov_reg, eigvals, eigvecs = regularize_cov_2d(cov)
    del cov_reg
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


class RobustEWCov2D:
    """Robust EW covariance recursion for 2D state coordinates."""

    def __init__(
        self,
        half_life: int = 78,
        c_huber: float = 2.5,
        eps_abs: float = 1e-8,
        eps_rel: float = 1e-4,
        warmup_weeks: int = 260,
    ) -> None:
        self.half_life = half_life
        self.c_huber = c_huber
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.warmup_weeks = warmup_weeks
        self.rho = 2 ** (-1 / half_life)

    def initialize_from_history(self, x_hist: np.ndarray) -> CovarianceState2D:
        """Initialize from static sample covariance plus `eps_abs * I`.

        Input:
            x_hist: Array with shape `(N, 2)`, `N >= 2`.
        Output:
            Covariance state with warmup_count set to 0.
        Time semantics:
            Caller must pass only history knowable at initialization time.
        """

        x = np.asarray(x_hist, dtype=float)
        if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] < 2:
            raise AssertionError("Need >= 2 observations with shape (N, 2)")
        if np.any(~np.isfinite(x)):
            raise ValueError("x_hist must be finite")
        mean_init = x.mean(axis=0)
        cov_sample = np.cov(x.T, ddof=1)
        cov_raw_init = cov_sample + self.eps_abs * np.eye(2)
        cov_reg_init, eigvals, eigvecs, diagnostics = regularize_cov_2d(
            cov_raw_init, self.eps_abs, self.eps_rel, return_diagnostics=True
        )
        diagnostics.update(
            {
                "maha": 0.0,
                "huber_weight": 1.0,
                "state_ok": True,
            }
        )
        return CovarianceState2D(
            mean=mean_init,
            cov_raw=cov_raw_init,
            cov_reg=cov_reg_init,
            eigvals=eigvals,
            eigvecs=eigvecs,
            warmup_count=0,
            last_diagnostics=diagnostics,
        )

    def update(self, state: CovarianceState2D, x_t: np.ndarray) -> CovarianceState2D:
        """Update state, skipping NaN numerics but preserving elapsed decay.

        If `x_t` contains NaN, numeric arrays are returned unchanged and
        `pending_missing_steps` increments. The next finite observation applies
        `rho ** (pending_missing_steps + 1)` and the complement weight.
        """

        x = np.asarray(x_t, dtype=float)
        if x.shape != (2,):
            raise ValueError("x_t must have shape (2,)")
        if np.any(~np.isfinite(x)):
            return CovarianceState2D(
                mean=state.mean.copy(),
                cov_raw=state.cov_raw.copy(),
                cov_reg=state.cov_reg.copy(),
                eigvals=state.eigvals.copy(),
                eigvecs=state.eigvecs.copy(),
                warmup_count=state.warmup_count,
                pending_missing_steps=state.pending_missing_steps + 1,
                last_diagnostics=state.last_diagnostics,
                state_ok=state.state_ok,
            )

        try:
            k = state.pending_missing_steps + 1
            effective_rho = self.rho**k
            delta = x - state.mean
            inv_cov = np.linalg.inv(state.cov_reg)
            maha = float(np.sqrt(delta.T @ inv_cov @ delta))
            huber_weight = min(1.0, self.c_huber / max(maha, 1e-12))
            delta_tilde = huber_weight * delta
            mean_new = effective_rho * state.mean + (1 - effective_rho) * x
            cov_raw_new = effective_rho * state.cov_raw + (
                1 - effective_rho
            ) * np.outer(delta_tilde, delta_tilde)
            cov_reg_new, eigvals_new, eigvecs_new, diagnostics = regularize_cov_2d(
                cov_raw_new,
                self.eps_abs,
                self.eps_rel,
                return_diagnostics=True,
            )
            diagnostics.update(
                {
                    "maha": maha,
                    "huber_weight": huber_weight,
                    "state_ok": True,
                    "elapsed_steps": k,
                    "effective_rho": effective_rho,
                }
            )
            LOGGER.info("covariance_update", extra={"diagnostics": diagnostics})
            return CovarianceState2D(
                mean=mean_new,
                cov_raw=cov_raw_new,
                cov_reg=cov_reg_new,
                eigvals=eigvals_new,
                eigvecs=eigvecs_new,
                warmup_count=state.warmup_count + 1,
                pending_missing_steps=0,
                last_diagnostics=diagnostics,
                state_ok=True,
            )
        except (np.linalg.LinAlgError, ValueError) as exc:
            diagnostics = {
                "maha": np.nan,
                "huber_weight": np.nan,
                "eigval_1": np.nan,
                "eigval_2_raw": np.nan,
                "eigval_2_reg": np.nan,
                "condition_number_raw": np.inf,
                "condition_number_reg": np.inf,
                "eigval_2_was_floored": False,
                "state_ok": False,
                "error": str(exc),
            }
            LOGGER.exception("covariance_update_failed", extra={"diagnostics": diagnostics})
            return CovarianceState2D(
                mean=state.mean.copy(),
                cov_raw=state.cov_raw.copy(),
                cov_reg=state.cov_reg.copy(),
                eigvals=state.eigvals.copy(),
                eigvecs=state.eigvecs.copy(),
                warmup_count=state.warmup_count,
                pending_missing_steps=state.pending_missing_steps,
                last_diagnostics=diagnostics,
                state_ok=False,
            )

    def is_warm(self, state: CovarianceState2D) -> bool:
        """Return whether state outputs are unlocked after warmup."""

        return state.warmup_count >= self.warmup_weeks
