from __future__ import annotations

from pathlib import Path

import pytest

from qqq_cycle.portfolio.policy import load_portfolio_policy
from qqq_cycle.portfolio.signal_gate import evaluate_signal_eligibility
from qqq_cycle.portfolio.target_weights import generate_target_weights


POLICY_PATH = Path("configs/portfolio_policy_v1.yaml")


def _strict_snapshot(rho_t: float) -> dict:
    return {
        "week_end": "2026-04-24",
        "mode": "strict",
        "backfill_mode": "strict_recovery",
        "strict_gate_passed": True,
        "execution_permitted": True,
        "h_t": 0.22,
        "rho_t": rho_t,
        "k_hat_t": 2,
        "s_t": 0.12,
    }


def _degraded_snapshot() -> dict:
    return {
        "week_end": "2026-04-24",
        "mode": "strict",
        "backfill_mode": "degraded_backfill",
        "strict_gate_passed": True,
        "execution_permitted": True,
        "h_t": 0.22,
        "rho_t": 0.34,
        "k_hat_t": 2,
        "s_t": 0.12,
    }


def test_degraded_signal_holds_prior_target() -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    gate = evaluate_signal_eligibility(
        _degraded_snapshot(),
        paper_only=True,
        broker_submission_allowed=False,
    )

    result = generate_target_weights(
        _degraded_snapshot(),
        gate,
        policy,
        prior_target_weights={"QQQ": 0.6, "BIL": 0.4},
    )

    assert result.generation_mode == "hold_prior_or_policy_default"
    assert result.target_weights == {"QQQ": 0.6, "BIL": 0.4}
    assert result.reason == "degraded_backfill_signal"


def test_degraded_signal_without_prior_uses_bil_default() -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    gate = evaluate_signal_eligibility(
        _degraded_snapshot(),
        paper_only=True,
        broker_submission_allowed=False,
    )

    result = generate_target_weights(_degraded_snapshot(), gate, policy, prior_target_weights=None)

    assert result.target_weights == {"QQQ": 0.0, "BIL": 1.0}
    assert result.reason == "no_prior_target_defensive_cash_proxy"


@pytest.mark.parametrize(
    ("rho_t", "bucket", "weights"),
    [
        (0.34, "risk_low", {"QQQ": 1.0, "BIL": 0.0}),
        (0.35, "risk_mid", {"QQQ": 0.6, "BIL": 0.4}),
        (0.65, "risk_high", {"QQQ": 0.2, "BIL": 0.8}),
    ],
)
def test_strict_rho_bucket_mapping(rho_t: float, bucket: str, weights: dict[str, float]) -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    snapshot = _strict_snapshot(rho_t)
    gate = evaluate_signal_eligibility(
        snapshot,
        paper_only=True,
        broker_submission_allowed=False,
    )

    result = generate_target_weights(snapshot, gate, policy)

    assert result.rho_bucket == bucket
    assert result.target_weights == weights
    assert abs(sum(result.target_weights.values()) - 1.0) < 1e-9


def test_unknown_asset_in_prior_target_is_rejected() -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    gate = evaluate_signal_eligibility(
        _degraded_snapshot(),
        paper_only=True,
        broker_submission_allowed=False,
    )

    with pytest.raises(ValueError):
        generate_target_weights(
            _degraded_snapshot(),
            gate,
            policy,
            prior_target_weights={"QQQ": 0.5, "TLT": 0.5},
        )


def test_discrete_bucket_turnover_risk_is_reported() -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    snapshot = _strict_snapshot(0.35)
    gate = evaluate_signal_eligibility(
        snapshot,
        paper_only=True,
        broker_submission_allowed=False,
    )

    result = generate_target_weights(snapshot, gate, policy)

    assert "discrete_bucket_turnover_risk" in result.known_limitations
