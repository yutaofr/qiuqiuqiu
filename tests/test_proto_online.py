import numpy as np

from qqq_cycle.core.covariance import symmetric_inv_sqrt_2d, symmetric_sqrt_2d
from qqq_cycle.core.proto_online import (
    PrototypeState,
    ReactivationBuffer,
    assign_cluster,
    batch_reactivation_update,
    initialize_prototypes_from_history,
    update_prototypes,
)


def test_assignment_uses_prior_covariance_geometry() -> None:
    prototypes = np.array([[0.0, 10.0], [3.0, 0.0]])
    theta = np.array([0.0, 0.0])
    cov_prior = np.array([[100.0, 0.0], [0.0, 1.0]])

    assigned, distances = assign_cluster(theta, prototypes, cov_prior)

    assert assigned == 1
    assert distances[1] < distances[0]


def test_assigned_cluster_updates_and_inactive_preserves_whitened_residual() -> None:
    mean = np.array([0.0, 0.0])
    cov_prev = np.array([[4.0, 0.0], [0.0, 1.0]])
    xi = np.array([[0.0, 0.0], [1.0, -1.0]])
    state = PrototypeState(
        centroids=np.array([[0.0, 0.0], [2.0, -1.0]]),
        weights=np.array([1.0, 1.0]),
        residuals=xi,
        last_active=np.array([0, 0]),
        reactivation_buffers=[ReactivationBuffer(), ReactivationBuffer()],
    )

    updated = update_prototypes(
        state=state,
        theta_t=np.array([1.0, 0.0]),
        mean_t=mean,
        cov_prev=cov_prev,
        cov_current=cov_prev,
        t=1,
    )

    expected_inactive = mean + symmetric_sqrt_2d(cov_prev) @ xi[1]
    np.testing.assert_allclose(updated.state.centroids[1], expected_inactive)
    assert updated.assigned_cluster == 0
    assert updated.state.weights[0] > state.weights[0] * updated.rho_mu


def test_residuals_stored_using_current_inverse_sqrt() -> None:
    mean = np.array([0.0, 0.0])
    cov_prev = np.eye(2)
    cov_current = np.array([[4.0, 0.0], [0.0, 9.0]])
    state = initialize_prototypes_from_history(
        np.array([[-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    )

    updated = update_prototypes(
        state=state,
        theta_t=np.array([2.0, 0.0]),
        mean_t=mean,
        cov_prev=cov_prev,
        cov_current=cov_current,
        t=10,
    ).state

    expected = (symmetric_inv_sqrt_2d(cov_current) @ (updated.centroids[4] - mean))
    np.testing.assert_allclose(updated.residuals[4], expected)


def test_reactivation_batch_update_equals_explicit_ew_replay() -> None:
    rho = 2 ** (-1 / 78)
    stale_mu = np.array([10.0, -3.0])
    stale_weight = 2.0
    gap = 7
    samples = np.array(
        [[1.0, 1.0], [2.0, 1.5], [3.0, 1.75], [4.0, 2.0]], dtype=float
    )

    batch_mu, batch_weight = batch_reactivation_update(
        stale_mu=stale_mu,
        stale_weight=stale_weight,
        gap_weeks=gap,
        samples_oldest_to_newest=samples,
        rho_mu=rho,
    )

    weight = rho**gap * stale_weight
    mu = stale_mu.copy()
    for sample in samples:
        weight_new = rho * weight + 1.0
        mu = (rho * weight * mu + sample) / weight_new
        weight = weight_new

    np.testing.assert_allclose(batch_mu, mu)
    np.testing.assert_allclose(batch_weight, weight)
