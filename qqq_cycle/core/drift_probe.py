"""Physical-space drift probe for replay diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qqq_cycle.core.state_layer import realized_vol_20w


class RollingPercentile:
    """Rolling empirical percentile with expanding cold start.

    The value at t is the share of values in the current historical window
    less than or equal to x_t, including x_t and excluding future rows.
    """

    def __init__(self, window: int = 520) -> None:
        self.window = window

    def transform(self, series: pd.Series) -> pd.Series:
        values = pd.Series(series, index=series.index, dtype=float)
        out = np.full(len(values), np.nan)
        arr = values.to_numpy()
        for i, x in enumerate(arr):
            if not np.isfinite(x):
                continue
            start = max(0, i - self.window + 1)
            window_values = arr[start : i + 1]
            window_values = window_values[np.isfinite(window_values)]
            if len(window_values) == 0:
                continue
            out[i] = float(np.sum(window_values <= x) / len(window_values))
        return pd.Series(out, index=values.index)


class DriftProbe:
    """Compute physical-space drift degree and drift flag.

    This is the minimum implementation needed for replay diagnostics. It uses
    the v2.2 physical-space percentile mapping and a 520-week rolling median/MAD
    baseline with expanding cold start.
    """

    def __init__(
        self,
        pct_window: int = 520,
        ew_half_life: int = 260,
        theta_lo: float = 1.2,
        theta_hi: float = 1.8,
        eps: float = 1e-12,
    ) -> None:
        self.pct_window = pct_window
        self.ew_half_life = ew_half_life
        self.theta_lo = theta_lo
        self.theta_hi = theta_hi
        self.eps = eps

    def _rolling_mad_scale(self, series: pd.Series) -> pd.Series:
        out = np.full(len(series), np.nan)
        arr = series.to_numpy(dtype=float)
        for i in range(len(arr)):
            start = max(0, i - self.pct_window + 1)
            vals = arr[start : i + 1]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            med = np.median(vals)
            out[i] = 1.4826 * np.median(np.abs(vals - med))
        return pd.Series(out, index=series.index)

    def _ew_mean(self, series: pd.Series) -> pd.Series:
        rho = 2 ** (-1 / self.ew_half_life)
        out = np.full(len(series), np.nan)
        prev = np.nan
        for i, x in enumerate(series.to_numpy(dtype=float)):
            if not np.isfinite(x):
                out[i] = prev
                continue
            prev = x if not np.isfinite(prev) else rho * prev + (1 - rho) * x
            out[i] = prev
        return pd.Series(out, index=series.index)

    def compute(self, raw_inputs: pd.DataFrame) -> pd.DataFrame:
        """Return drift probe table indexed by decision week.

        Required columns: DFII10, DGS2, BAMLH0A0HYM2, NFCI, VIXCLS, QQQ.
        """

        pct = RollingPercentile(self.pct_window)
        qqq = raw_inputs["QQQ"].astype(float)
        dgs2_delta4 = raw_inputs["DGS2"].astype(float).diff(4)
        u1 = qqq / qqq.rolling(52, min_periods=52).mean() - 1.0
        u2 = qqq / qqq.rolling(156, min_periods=156).mean() - 1.0
        rv = realized_vol_20w(qqq)
        ma40_dev = qqq / qqq.rolling(40, min_periods=40).mean() - 1.0

        l_raw = 0.25 * (
            -pct.transform(raw_inputs["DFII10"])
            - pct.transform(dgs2_delta4)
            - pct.transform(raw_inputs["BAMLH0A0HYM2"])
            - pct.transform(raw_inputs["NFCI"])
        )
        t_raw = 0.5 * (pct.transform(u1) + pct.transform(u2))
        p_raw = (1.0 / 3.0) * (
            -pct.transform(raw_inputs["VIXCLS"]) - pct.transform(rv) + pct.transform(ma40_dev)
        )
        h_raw = 0.40 * l_raw + 0.35 * t_raw + 0.25 * p_raw
        h_bar = self._ew_mean(h_raw)
        h_med = h_raw.rolling(self.pct_window, min_periods=1).median()
        h_scale = self._rolling_mad_scale(h_raw)
        drift_raw = (h_bar - h_med) / (h_scale + self.eps)
        drift_flag = (drift_raw.abs() >= self.theta_hi).astype("Int64")
        return pd.DataFrame(
            {
                "H_raw": h_raw,
                "H_bar_raw": h_bar,
                "drift_probe_raw": drift_raw,
                "drift_flag": drift_flag,
            },
            index=raw_inputs.index,
        )
