"""Rolling quantile helpers for dynamic 2D noise floors."""

from __future__ import annotations

import numpy as np


def rolling_quantile_diag_2d(
    delta_v: np.ndarray,
    window: int = 520,
    quantile: float = 0.10,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return diagonal quantile matrices from raw 2D velocity increments.

    Input:
        delta_v: Array of shape `(T, 2)` containing raw velocity increments,
            not pre-squared values. The function squares these increments
            internally before quantile extraction.
        window: Rolling window length; rows before `window` use expanding history.
        quantile: Quantile applied independently to each squared coordinate.
        eps: Lower floor applied to diagonal entries.
    Output:
        Array of shape `(T, 2, 2)` containing diagonal matrices from
        rolling/expanding quantiles of squared increments. Off-diagonal
        entries are fixed at zero.
    Time semantics:
        Row t uses only rows `max(0, t-window+1):t`, inclusive. No future rows
        are used, and NaNs are ignored coordinate-wise.
    """

    values = np.asarray(delta_v, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("delta_v must have shape (T, 2)")
    if window < 1:
        raise ValueError("window must be >= 1")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")

    out = np.zeros((len(values), 2, 2), dtype=float)
    squared = values**2
    for i in range(len(values)):
        start = max(0, i - window + 1)
        current = squared[start : i + 1]
        diag = np.empty(2, dtype=float)
        for col in range(2):
            col_values = current[:, col]
            col_values = col_values[np.isfinite(col_values)]
            diag[col] = eps if len(col_values) == 0 else max(
                float(np.quantile(col_values, quantile)), eps
            )
        out[i] = np.diag(diag)
    return out
