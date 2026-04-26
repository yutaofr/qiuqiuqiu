import numpy as np
import pandas as pd

from qqq_cycle.core.drift_probe import DriftProbe, RollingPercentile


def test_rolling_percentile_uses_expanding_then_rolling_window() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 2.0], index=pd.date_range("2020-01-01", periods=4))
    pct = RollingPercentile(window=3).transform(series)

    np.testing.assert_allclose(pct.to_numpy(), [1.0, 1.0, 1.0, 2.0 / 3.0])


def test_drift_probe_outputs_raw_delta_and_flag() -> None:
    idx = pd.date_range("2020-01-03", periods=220, freq="W-FRI")
    values = np.r_[np.zeros(180), np.ones(40) * 10.0]
    raw = pd.DataFrame(
        {
            "DFII10": values,
            "DGS2": values,
            "BAMLH0A0HYM2": values,
            "NFCI": values,
            "VIXCLS": values,
            "QQQ": 100.0 + np.arange(220),
        },
        index=idx,
    )

    out = DriftProbe(pct_window=20, ew_half_life=10, theta_hi=0.5).compute(raw)

    assert {"drift_probe_raw", "drift_flag"}.issubset(out.columns)
    assert out["drift_probe_raw"].notna().any()
    assert out["drift_flag"].isin([0, 1]).all()
