from __future__ import annotations

from pathlib import Path

import pytest

from qqq_cycle.portfolio.delta import build_portfolio_delta
from qqq_cycle.portfolio.policy import load_portfolio_policy
from qqq_cycle.portfolio.portfolio_snapshot import load_portfolio_snapshot
from qqq_cycle.portfolio.signal_gate import evaluate_signal_eligibility
from qqq_cycle.portfolio.target_weights import generate_target_weights


POLICY_PATH = Path("configs/portfolio_policy_v1.yaml")


def _snapshot_csv() -> str:
    return (
        "account_id,week_end,symbol,quantity,market_price,market_value,weight,cash,source,paper_only,broker_submission_allowed\n"
        "acct,2026-04-24,QQQ,1.2,500.0,600.0,0.6,0.0,test,true,false\n"
        "acct,2026-04-24,BIL,4.0,100.0,400.0,0.4,0.0,test,true,false\n"
    )


def _strict_snapshot(rho_t: float) -> dict:
    return {
        "week_end": "2026-04-24",
        "mode": "strict",
        "backfill_mode": "strict_recovery",
        "strict_gate_passed": True,
        "execution_permitted": True,
        "h_t": 0.2,
        "rho_t": rho_t,
        "k_hat_t": 2,
        "s_t": 0.1,
    }


def test_turnover_below_threshold_suppresses_rebalance(tmp_path: Path) -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    snapshot_path = tmp_path / "snapshot.csv"
    snapshot_path.write_text(_snapshot_csv(), encoding="utf-8")
    portfolio = load_portfolio_snapshot(snapshot_path, policy)
    gate = evaluate_signal_eligibility(_strict_snapshot(0.35), paper_only=True, broker_submission_allowed=False)
    target = generate_target_weights(_strict_snapshot(0.35), gate, policy)

    delta = build_portfolio_delta(portfolio, target, gate, policy)

    assert delta.turnover == pytest.approx(0.0)
    assert delta.rebalance_required is False


def test_turnover_above_threshold_requires_rebalance(tmp_path: Path) -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    snapshot_path = tmp_path / "snapshot.csv"
    snapshot_path.write_text(_snapshot_csv(), encoding="utf-8")
    portfolio = load_portfolio_snapshot(snapshot_path, policy)
    gate = evaluate_signal_eligibility(_strict_snapshot(0.65), paper_only=True, broker_submission_allowed=False)
    target = generate_target_weights(_strict_snapshot(0.65), gate, policy)

    delta = build_portfolio_delta(portfolio, target, gate, policy)

    assert delta.turnover == pytest.approx(0.4)
    assert delta.rebalance_required is True


def test_execution_not_allowed_suppresses_rebalance(tmp_path: Path) -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    snapshot_path = tmp_path / "snapshot.csv"
    snapshot_path.write_text(_snapshot_csv(), encoding="utf-8")
    portfolio = load_portfolio_snapshot(snapshot_path, policy)
    degraded = {
        "week_end": "2026-04-24",
        "mode": "strict",
        "backfill_mode": "degraded_backfill",
        "strict_gate_passed": True,
        "execution_permitted": True,
        "h_t": 0.2,
        "rho_t": 0.65,
        "k_hat_t": 2,
        "s_t": 0.1,
    }
    gate = evaluate_signal_eligibility(degraded, paper_only=True, broker_submission_allowed=False)
    target = generate_target_weights(degraded, gate, policy, prior_target_weights=portfolio.weights)

    delta = build_portfolio_delta(portfolio, target, gate, policy)

    assert delta.rebalance_required is False
    assert delta.reason == "degraded_backfill_signal"


def test_unknown_current_symbol_rejected(tmp_path: Path) -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    snapshot_path = tmp_path / "snapshot.csv"
    snapshot_path.write_text(_snapshot_csv().replace("BIL", "TLT"), encoding="utf-8")

    with pytest.raises(ValueError):
        load_portfolio_snapshot(snapshot_path, policy)
