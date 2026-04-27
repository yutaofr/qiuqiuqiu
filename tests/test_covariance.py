import numpy as np

from qqq_cycle.core.covariance import COND_WARN_RATIO, RobustEWCov2D, regularize_cov_2d


def test_cold_start_invertible() -> None:
    cov = RobustEWCov2D()
    state = cov.initialize_from_history(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.5]]))

    np.linalg.inv(state.cov_reg)
    assert state.cov_reg.shape == (2, 2)


def test_warmup_gate() -> None:
    cov = RobustEWCov2D(warmup_weeks=3)
    state = cov.initialize_from_history(np.array([[0.0, 0.0], [1.0, 1.0]]))

    assert not cov.is_warm(state)
    for i in range(3):
        state = cov.update(state, np.array([float(i), float(i + 1)]))
    assert cov.is_warm(state)


def test_nan_input_preserves_numeric_state_and_accumulates_elapsed_decay() -> None:
    cov = RobustEWCov2D(half_life=10)
    state = cov.initialize_from_history(np.array([[0.0, 0.0], [1.0, 1.0]]))
    mean_prev = state.mean.copy()
    cov_prev = state.cov_raw.copy()

    skipped = cov.update(state, np.array([np.nan, 2.0]))

    np.testing.assert_allclose(skipped.mean, mean_prev)
    np.testing.assert_allclose(skipped.cov_raw, cov_prev)
    assert skipped.pending_missing_steps == 1


def test_next_valid_update_uses_effective_rho_complement() -> None:
    cov = RobustEWCov2D(half_life=10, c_huber=1e9)
    state = cov.initialize_from_history(np.array([[0.0, 0.0], [1.0, 1.0]]))
    skipped = cov.update(state, np.array([np.nan, 2.0]))
    x = np.array([2.0, 3.0])
    updated = cov.update(skipped, x)

    rho = 2 ** (-1 / 10)
    effective_rho = rho**2
    expected_mean = effective_rho * state.mean + (1 - effective_rho) * x
    delta = x - state.mean
    expected_cov = effective_rho * state.cov_raw + (1 - effective_rho) * np.outer(
        delta, delta
    )

    np.testing.assert_allclose(updated.mean, expected_mean)
    np.testing.assert_allclose(updated.cov_raw, expected_cov)
    assert updated.pending_missing_steps == 0


def test_huber_truncation_bounds_outlier() -> None:
    cov = RobustEWCov2D(c_huber=2.5)
    state = cov.initialize_from_history(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))

    updated = cov.update(state, np.array([100.0, 100.0]))

    assert updated.last_diagnostics is not None
    assert updated.last_diagnostics["huber_weight"] < 1.0


def test_eigval_floored_flag_and_condition_diagnostics() -> None:
    cov_raw = np.array([[1.0, 0.0], [0.0, 1e-12]])

    _, eigvals_reg, _, diagnostics = regularize_cov_2d(
        cov_raw, eps_abs=1e-8, eps_rel=1e-4, return_diagnostics=True
    )

    assert eigvals_reg[1] == 1e-4
    assert diagnostics["eigval_2_was_floored"] is True
    assert diagnostics["condition_number_raw"] > diagnostics["condition_number_reg"]


def test_regularize_cov_2d_returns_three_values_unless_diagnostics_requested() -> None:
    cov_raw = np.array([[1.0, 0.1], [0.1, 0.5]])

    plain = regularize_cov_2d(cov_raw)
    with_diagnostics = regularize_cov_2d(cov_raw, return_diagnostics=True)

    assert len(plain) == 3
    assert len(with_diagnostics) == 4
    assert isinstance(with_diagnostics[3], dict)


def test_condition_threshold_reachable() -> None:
    eps_rel = 1e-4
    cov_raw = np.array([[1.0, 0.0], [0.0, 1e-12]])

    _, _, _, diagnostics = regularize_cov_2d(
        cov_raw,
        eps_abs=1e-12,
        eps_rel=eps_rel,
        return_diagnostics=True,
    )

    assert diagnostics["condition_number_reg"] > COND_WARN_RATIO / eps_rel
