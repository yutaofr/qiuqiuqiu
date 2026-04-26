import numpy as np
import pandas as pd

from qqq_cycle.core.dual_memory import (
    DualMemoryNormalizer,
    dual_memory,
    exo_dual_memory,
    z_ew,
    z_ew_exo_with_huber_var,
    z_rob,
)


def test_robust_window_excludes_today_and_future() -> None:
    idx = pd.date_range("2020-01-03", periods=8, freq="W-FRI")
    base = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=idx, dtype=float)
    changed_future = base.copy()
    changed_future.iloc[-1] = 10_000.0

    out_base = z_rob(base, window=3, eps=1e-12)
    out_changed = z_rob(changed_future, window=3, eps=1e-12)

    assert out_base.iloc[6] == out_changed.iloc[6]


def test_z_ew_warmup_nan() -> None:
    series = pd.Series(np.arange(20, dtype=float))

    out = z_ew(series, half_life=8, eps=1e-12)

    assert out.iloc[0:2].isna().all()
    assert np.isfinite(out.iloc[2])


def test_dual_memory_nan_policy() -> None:
    series = pd.Series(np.arange(20, dtype=float))

    out = dual_memory(series, robust_window=5, ew_half_life=8, eps=1e-12)

    assert out.iloc[:5].isna().all()
    assert np.isfinite(out.iloc[6])


def test_exo_clip_bounded() -> None:
    series = pd.Series(np.r_[np.ones(270), [1e12]])

    out = exo_dual_memory(series, eps=1e-12)

    assert out.dropna().between(-5.0, 5.0).all()


def test_exo_huber_var_not_polluted_by_single_spike() -> None:
    base = pd.Series(np.ones(340))
    shocked = base.copy()
    shocked.iloc[280] = 1e12

    out_base = z_ew_exo_with_huber_var(base, half_life=260, huber_k=4.0, eps=1e-12)
    out_shocked = z_ew_exo_with_huber_var(
        shocked, half_life=260, huber_k=4.0, eps=1e-12
    )

    assert abs(out_shocked.iloc[320] - out_base.iloc[320]) < 0.1


def test_incremental_transform_matches_batch_next_point() -> None:
    history = pd.Series(np.linspace(1.0, 4.0, 40))
    x_new = 4.1
    normalizer = DualMemoryNormalizer(robust_window=8, ew_half_life=8)

    incremental = normalizer.transform_incremental(x_new, history)
    batch = normalizer.fit_transform(pd.concat([history, pd.Series([x_new])])).iloc[-1]

    assert np.isclose(incremental, batch, rtol=1e-12, atol=1e-12)
