"""Online prototype assignment and centroid maintenance."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qqq_cycle.core.covariance import symmetric_inv_sqrt_2d, symmetric_sqrt_2d


@dataclass
class ReactivationBuffer:
    """Four-sample reactivation buffer for a stale cluster."""

    samples: list[np.ndarray] = field(default_factory=list)
    active: bool = False

    def add(self, sample: np.ndarray) -> None:
        self.active = True
        self.samples.append(np.asarray(sample, dtype=float))
        if len(self.samples) > 4:
            self.samples = self.samples[-4:]

    def ready(self) -> bool:
        return len(self.samples) >= 4

    def clear(self) -> None:
        self.samples.clear()
        self.active = False


@dataclass
class PrototypeState:
    """Five-cluster online prototype state."""

    centroids: np.ndarray
    weights: np.ndarray
    residuals: np.ndarray
    last_active: np.ndarray
    reactivation_buffers: list[ReactivationBuffer]


@dataclass
class PrototypeUpdateResult:
    """Result of one online prototype step."""

    state: PrototypeState
    assigned_cluster: int
    distances: np.ndarray
    rho_mu: float


def assign_cluster(
    theta_t: np.ndarray, centroids_prev: np.ndarray, cov_prev: np.ndarray
) -> tuple[int, np.ndarray]:
    """Assign one point using prior covariance geometry."""

    theta = np.asarray(theta_t, dtype=float)
    inv_cov = np.linalg.inv(cov_prev)
    diffs = centroids_prev - theta
    distances = np.einsum("ki,ij,kj->k", diffs, inv_cov, diffs)
    return int(np.argmin(distances)), distances


def initialize_prototypes_from_history(theta_hist: np.ndarray) -> PrototypeState:
    """Bootstrap five prototypes deterministically by H-sorted bins."""

    x = np.asarray(theta_hist, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] < 5:
        raise ValueError("theta_hist must have shape (N, 2), N >= 5")
    ordered = x[np.argsort(x[:, 0])]
    bins = np.array_split(ordered, 5)
    centroids = np.vstack([b.mean(axis=0) for b in bins])
    weights = np.array([len(b) for b in bins], dtype=float)
    mean = x.mean(axis=0)
    cov = np.cov(x.T, ddof=1) + 1e-8 * np.eye(2)
    inv_sqrt = symmetric_inv_sqrt_2d(cov)
    residuals = np.vstack([inv_sqrt @ (c - mean) for c in centroids])
    return PrototypeState(
        centroids=centroids,
        weights=weights,
        residuals=residuals,
        last_active=np.zeros(5, dtype=int),
        reactivation_buffers=[ReactivationBuffer() for _ in range(5)],
    )


def batch_reactivation_update(
    stale_mu: np.ndarray,
    stale_weight: float,
    gap_weeks: int,
    samples_oldest_to_newest: np.ndarray,
    rho_mu: float,
) -> tuple[np.ndarray, float]:
    """Closed batch update equivalent to replaying four EW centroid updates."""

    samples = np.asarray(samples_oldest_to_newest, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("samples must have shape (N, 2)")
    weight = (rho_mu**gap_weeks) * float(stale_weight)
    mu = np.asarray(stale_mu, dtype=float).copy()
    for sample in samples:
        weight_new = rho_mu * weight + 1.0
        mu = (rho_mu * weight * mu + sample) / weight_new
        weight = weight_new
    return mu, weight


def update_prototypes(
    state: PrototypeState,
    theta_t: np.ndarray,
    mean_t: np.ndarray,
    cov_prev: np.ndarray,
    cov_current: np.ndarray,
    t: int,
    *,
    rho_mu: float = 2 ** (-1 / 78),
    stale_gap_weeks: int = 26,
) -> PrototypeUpdateResult:
    """Run one point-in-time prototype assignment/update step.

    Assignment and inactive-cluster mapping use `cov_prev`. Residual storage
    after all centroid updates uses `cov_current`.
    """

    assigned, distances = assign_cluster(theta_t, state.centroids, cov_prev)
    centroids = state.centroids.copy()
    weights = state.weights.copy()
    last_active = state.last_active.copy()
    buffers = [ReactivationBuffer(list(b.samples), b.active) for b in state.reactivation_buffers]
    sqrt_prev = symmetric_sqrt_2d(cov_prev)

    for k in range(len(centroids)):
        if k == assigned:
            gap = max(0, t - int(last_active[k]))
            if gap >= stale_gap_weeks:
                buffers[k].add(theta_t)
                if buffers[k].ready():
                    centroids[k], weights[k] = batch_reactivation_update(
                        centroids[k],
                        weights[k],
                        gap,
                        np.asarray(buffers[k].samples),
                        rho_mu,
                    )
                    buffers[k].clear()
                    last_active[k] = t
                else:
                    weights[k] = rho_mu * weights[k]
            else:
                weight_new = rho_mu * weights[k] + 1.0
                centroids[k] = (rho_mu * weights[k] * centroids[k] + theta_t) / weight_new
                weights[k] = weight_new
                last_active[k] = t
        else:
            weights[k] = rho_mu * weights[k]
            centroids[k] = mean_t + sqrt_prev @ state.residuals[k]

    inv_sqrt_current = symmetric_inv_sqrt_2d(cov_current)
    residuals = np.vstack([inv_sqrt_current @ (c - mean_t) for c in centroids])
    return PrototypeUpdateResult(
        state=PrototypeState(
            centroids=centroids,
            weights=weights,
            residuals=residuals,
            last_active=last_active,
            reactivation_buffers=buffers,
        ),
        assigned_cluster=assigned,
        distances=distances,
        rho_mu=rho_mu,
    )
