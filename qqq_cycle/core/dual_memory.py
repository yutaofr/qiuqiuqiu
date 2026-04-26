"""Dual-memory normalization from the QQQ cycle-state specification."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_float_series(series: pd.Series) -> pd.Series:
    return pd.Series(series, index=series.index, dtype=float)


def z_rob(series: pd.Series, window: int, eps: float = 1e-12) -> pd.Series:
    """Short-memory robust z-score using only `t-w:t-1` history."""

    s = _as_float_series(series)
    shifted = s.shift(1)
    med = shifted.rolling(window, min_periods=window).median()
    mad = shifted.rolling(window, min_periods=window).apply(
        lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True
    )
    return (s - med) / (1.4826 * mad + eps)


def z_ew(series: pd.Series, half_life: int, eps: float = 1e-12) -> pd.Series:
    """Long-memory EW z-score with explicit warmup NaNs."""

    s = _as_float_series(series)
    rho = 2 ** (-1 / half_life)
    n = len(s)
    out = np.full(n, np.nan)
    if n == 0:
        return pd.Series(out, index=s.index)
    values = s.to_numpy(dtype=float)
    mu = np.full(n, np.nan)
    var = np.full(n, np.nan)
    mu[0] = values[0]
    var[0] = 0.0
    min_ew_warmup = max(2, half_life // 4)
    for i in range(1, n):
        x = values[i]
        if np.isnan(x) or np.isnan(mu[i - 1]):
            mu[i] = mu[i - 1]
            var[i] = var[i - 1]
            continue
        mu[i] = rho * mu[i - 1] + (1 - rho) * x
        var[i] = rho * var[i - 1] + (1 - rho) * (x - mu[i - 1]) ** 2
        if i >= min_ew_warmup:
            out[i] = (x - mu[i]) / np.sqrt(var[i] + eps)
    return pd.Series(out, index=s.index)


def dual_memory(
    series: pd.Series, robust_window: int = 104, ew_half_life: int = 260, eps: float = 1e-12
) -> pd.Series:
    """Equal-weight robust/EW dual-memory normalization."""

    return 0.5 * z_rob(series, robust_window, eps) + 0.5 * z_ew(
        series, ew_half_life, eps
    )


def exo_pretransform(series: pd.Series) -> pd.Series:
    """Apply exogenous log1p pre-transform after lower clipping at zero."""

    s = _as_float_series(series)
    return pd.Series(np.log1p(s.clip(lower=0.0)), index=s.index)


def z_ew_exo_with_huber_var(
    series: pd.Series, half_life: int = 260, huber_k: float = 4.0, eps: float = 1e-12
) -> pd.Series:
    """EW z-score for exogenous variables with Huber clipping on variance only."""

    s = _as_float_series(series)
    rho = 2 ** (-1 / half_life)
    n = len(s)
    out = np.full(n, np.nan)
    if n == 0:
        return pd.Series(out, index=s.index)
    values = s.to_numpy(dtype=float)
    mu = np.full(n, np.nan)
    var = np.full(n, np.nan)
    mu[0] = values[0]
    var[0] = 0.0
    min_ew_warmup = max(2, half_life // 4)
    for i in range(1, n):
        x = values[i]
        if np.isnan(x) or np.isnan(mu[i - 1]):
            mu[i] = mu[i - 1]
            var[i] = var[i - 1]
            continue
        sigma_prev = np.sqrt(var[i - 1] + eps)
        delta = x - mu[i - 1]
        delta_clip = np.clip(delta, -huber_k * sigma_prev, huber_k * sigma_prev)
        mu[i] = rho * mu[i - 1] + (1 - rho) * x
        var[i] = rho * var[i - 1] + (1 - rho) * delta_clip**2
        if i >= min_ew_warmup:
            out[i] = (x - mu[i]) / np.sqrt(var[i] + eps)
    return pd.Series(out, index=s.index)


def exo_dual_memory(series: pd.Series, eps: float = 1e-12) -> pd.Series:
    """Exogenous dual-memory score clipped to [-5, 5]."""

    transformed = exo_pretransform(series)
    z1 = z_rob(transformed, window=260, eps=eps)
    z2 = z_ew_exo_with_huber_var(
        transformed, half_life=260, huber_k=4.0, eps=eps
    )
    return pd.Series(np.clip(0.5 * z1 + 0.5 * z2, -5.0, 5.0), index=series.index)


class DualMemoryNormalizer:
    """State-light normalizer interface for batch and next-point transforms."""

    def __init__(
        self,
        robust_window: int,
        ew_half_life: int,
        eps: float = 1e-12,
        clip: tuple[float, float] | None = None,
        exo_var_huber_k: float | None = None,
    ) -> None:
        self.robust_window = robust_window
        self.ew_half_life = ew_half_life
        self.eps = eps
        self.clip = clip
        self.exo_var_huber_k = exo_var_huber_k

    def fit_transform(self, x: pd.Series) -> pd.Series:
        """Return batch transform using only each row's prior history."""

        if self.exo_var_huber_k is None:
            out = dual_memory(x, self.robust_window, self.ew_half_life, self.eps)
        else:
            s = exo_pretransform(x)
            out = 0.5 * z_rob(s, self.robust_window, self.eps) + 0.5 * z_ew_exo_with_huber_var(
                s, self.ew_half_life, self.exo_var_huber_k, self.eps
            )
        if self.clip is not None:
            out = pd.Series(np.clip(out, self.clip[0], self.clip[1]), index=x.index)
        return out

    def transform_incremental(self, x_new: float, history: pd.Series) -> float:
        """Transform one new value as if appended after `history`."""

        next_index = (
            history.index[-1] + 1
            if len(history) and not isinstance(history.index, pd.DatetimeIndex)
            else pd.Timestamp.max
        )
        appended = pd.concat([history, pd.Series([x_new], index=[next_index])])
        return float(self.fit_transform(appended).iloc[-1])
