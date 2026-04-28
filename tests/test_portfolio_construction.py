"""Tests for Phase 11 portfolio construction rules."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from qqq_cycle.portfolio.construction import (
    BacktestConfig,
    apply_circuit_breaker,
    apply_turnover_threshold,
    build_weekly_weights,
    map_rho_to_target_weights,
)


def test_rho_linear_mapping_to_weights() -> None:
    omega_qqq, omega_shy = map_rho_to_target_weights(0.3)

    assert omega_qqq == pytest.approx(0.7)
    assert omega_shy == pytest.approx(0.3)


@pytest.mark.parametrize("rho_t", [0.0, 0.2, 0.5, 0.9, 1.0])
def test_weights_sum_to_one(rho_t: float) -> None:
    omega_qqq, omega_shy = map_rho_to_target_weights(rho_t)

    assert omega_qqq >= 0.0
    assert omega_shy >= 0.0
    assert omega_qqq + omega_shy == pytest.approx(1.0)


def test_turnover_threshold_blocks_small_rebalance() -> None:
    omega_qqq, required = apply_turnover_threshold(0.50, 0.54, threshold=0.05)

    assert omega_qqq == pytest.approx(0.50)
    assert required is False


def test_circuit_breaker_forces_full_shy() -> None:
    signal_df = pd.DataFrame(
        {
            "week_end": ["2024-01-05"],
            "rho_t": [0.2],
            "k_hat_t": [0],
            "drift_flag": [1],
        }
    )
    config = BacktestConfig(circuit_breaker_s1_index=0)

    weights = build_weekly_weights(signal_df, config)

    assert weights[0].omega_qqq_final == pytest.approx(0.0)
    assert weights[0].omega_shy_final == pytest.approx(1.0)
    assert weights[0].rebalance_required is True
    assert weights[0].circuit_breaker_active is True


def test_circuit_breaker_requires_two_weeks_to_release() -> None:
    active, outside = apply_circuit_breaker(
        k_hat_t=0,
        drift_flag=1,
        breaker_active=False,
        weeks_outside_s1=0,
        s1_index=0,
        release_weeks=2,
    )
    assert active is True
    assert outside == 0

    active, outside = apply_circuit_breaker(
        k_hat_t=1,
        drift_flag=0,
        breaker_active=active,
        weeks_outside_s1=outside,
        s1_index=0,
        release_weeks=2,
    )
    assert active is True
    assert outside == 1

    active, outside = apply_circuit_breaker(
        k_hat_t=2,
        drift_flag=0,
        breaker_active=active,
        weeks_outside_s1=outside,
        s1_index=0,
        release_weeks=2,
    )
    assert active is False
    assert outside == 2


def test_compute_s1_cluster_index_matches_manifest() -> None:
    from qqq_cycle.portfolio.construction import compute_s1_cluster_index

    pipeline_df = pd.DataFrame(
        {
            "k_hat_t": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            "interpretability": [
                json.dumps({"H": -2.0, "I": -1.0, "drift_flag": 0}),
                json.dumps({"H": -1.8, "I": -0.8, "drift_flag": 0}),
                json.dumps({"H": -1.5, "I": 0.5, "drift_flag": 0}),
                json.dumps({"H": -1.4, "I": 0.6, "drift_flag": 0}),
                json.dumps({"H": 0.1, "I": 0.0, "drift_flag": 0}),
                json.dumps({"H": 0.2, "I": 0.1, "drift_flag": 0}),
                json.dumps({"H": 0.4, "I": -0.3, "drift_flag": 0}),
                json.dumps({"H": 0.5, "I": -0.2, "drift_flag": 0}),
                json.dumps({"H": 0.6, "I": 0.4, "drift_flag": 0}),
                json.dumps({"H": 0.7, "I": 0.5, "drift_flag": 0}),
            ],
        }
    )

    assert compute_s1_cluster_index(pipeline_df) == 0
