"""Tests for the live pipeline script helpers."""

from __future__ import annotations

from scripts.run_live_pipeline import _print_result
from qqq_cycle.live.runtime import LiveRunResult


def test_print_result_handles_none_metrics(capsys) -> None:
    result = LiveRunResult(
        asof_week_end="2026-05-01",
        mode="degraded",
        execution_state="block",
        signal_bundle={
            "week_end": "2026-05-01",
            "mode": "degraded",
            "k_hat_t": None,
            "s_t": None,
            "h_t": None,
            "rho_t": None,
            "I_t": None,
        },
        portfolio_bundle={
            "omega_qqq_target": 0.5,
            "omega_shy_target": 0.5,
            "omega_qqq_final": 0.5,
            "omega_shy_final": 0.5,
            "rebalance_required": False,
            "circuit_breaker_active": False,
            "reason": "rho_t_missing",
        },
        interpretability_bundle={},
        state_path="state/live_state_latest",
        degraded_reason="h_t unavailable",
        execution_permitted=False,
        execution_block_reason="missing weights",
        signal_valid_but_not_executable=False,
        freshness_snapshot=[],
    )

    _print_result(result)
    output = capsys.readouterr().out
    assert "s_t=n/a" in output
    assert "h_t=n/a" in output
    assert "rho_t=n/a" in output
