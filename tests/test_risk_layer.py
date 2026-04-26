import numpy as np

from qqq_cycle.core.risk_layer import blended_state_weight


def test_blended_state_weight_uses_natural_weight_below_drift_band() -> None:
    p_t = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    omega = np.array([1.0, 0.7, 0.3, 0.6, 0.9])

    out = blended_state_weight(p_t, omega, delta_abs_raw=1.0)

    assert out == 1.0


def test_blended_state_weight_uses_neutral_above_drift_band() -> None:
    p_t = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    omega = np.array([1.0, 0.7, 0.3, 0.6, 0.9])

    out = blended_state_weight(p_t, omega, delta_abs_raw=-2.0)

    assert out == 0.6


def test_blended_state_weight_is_linear_inside_drift_band() -> None:
    p_t = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    omega = np.array([1.0, 0.7, 0.3, 0.6, 0.9])

    out = blended_state_weight(p_t, omega, delta_abs_raw=1.5)

    assert np.isclose(out, 0.8)
