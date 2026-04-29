from __future__ import annotations

import json
from pathlib import Path

from qqq_cycle.portfolio.delta import build_portfolio_delta
from qqq_cycle.portfolio.order_simulator import simulate_hypothetical_orders
from qqq_cycle.portfolio.policy import load_portfolio_policy
from qqq_cycle.portfolio.portfolio_snapshot import load_portfolio_snapshot
from qqq_cycle.portfolio.signal_gate import evaluate_signal_eligibility
from qqq_cycle.portfolio.target_weights import generate_target_weights


POLICY_PATH = Path("configs/portfolio_policy_v1.yaml")
REAL_PHASE14_PATH = Path("outputs/phase14/history/cycle_snapshot_2026-04-24__run_2026-04-29T10-16-27Z.json")


def _snapshot_csv() -> str:
    return (
        "account_id,week_end,symbol,quantity,market_price,market_value,weight,cash,source,paper_only,broker_submission_allowed\n"
        "acct,2026-04-24,QQQ,1.2,500.0,600.0,0.6,0.0,test,true,false\n"
        "acct,2026-04-24,BIL,4.0,100.0,400.0,0.4,0.0,test,true,false\n"
    )


def _strict_snapshot(tmp_path: Path, rho_t: float = 0.65) -> dict:
    path = tmp_path / "phase14_snapshot.json"
    payload = {
        "week_end": "2026-04-24",
        "mode": "strict",
        "backfill_mode": "strict_recovery",
        "strict_gate_passed": True,
        "execution_permitted": True,
        "h_t": 0.25,
        "rho_t": rho_t,
        "k_hat_t": 2,
        "s_t": 0.15,
        "source_hash": "synthetic",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def test_degraded_signal_generates_zero_orders(tmp_path: Path) -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    portfolio_path = tmp_path / "snapshot.csv"
    portfolio_path.write_text(_snapshot_csv(), encoding="utf-8")
    portfolio = load_portfolio_snapshot(portfolio_path, policy)
    degraded = json.loads(REAL_PHASE14_PATH.read_text(encoding="utf-8"))
    gate = evaluate_signal_eligibility(degraded, paper_only=True, broker_submission_allowed=False)
    target = generate_target_weights(degraded, gate, policy, prior_target_weights=portfolio.weights)
    delta = build_portfolio_delta(portfolio, target, gate, policy)

    result = simulate_hypothetical_orders(portfolio, delta, gate, policy)

    assert result.orders_count == 0
    assert result.reason == "degraded_backfill_signal"


def test_strict_synthetic_signal_can_generate_orders(tmp_path: Path) -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    portfolio_path = tmp_path / "snapshot.csv"
    portfolio_path.write_text(_snapshot_csv(), encoding="utf-8")
    portfolio = load_portfolio_snapshot(portfolio_path, policy)
    strict_snapshot = _strict_snapshot(tmp_path, rho_t=0.0)
    gate = evaluate_signal_eligibility(strict_snapshot, paper_only=True, broker_submission_allowed=False)
    target = generate_target_weights(strict_snapshot, gate, policy)
    delta = build_portfolio_delta(portfolio, target, gate, policy)

    result = simulate_hypothetical_orders(portfolio, delta, gate, policy)

    assert result.orders_count > 0
    assert all(order.paper_only is True for order in result.orders)
    assert all(order.broker_submission_allowed is False for order in result.orders)


def test_full_cash_target_with_commission_does_not_create_negative_cash(tmp_path: Path) -> None:
    policy_payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy_payload["execution_model"]["commission_per_order"] = 1.0
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(json.dumps(policy_payload), encoding="utf-8")
    policy = load_portfolio_policy(policy_path)

    portfolio_path = tmp_path / "snapshot.csv"
    portfolio_path.write_text(_snapshot_csv(), encoding="utf-8")
    portfolio = load_portfolio_snapshot(portfolio_path, policy)
    strict_snapshot = _strict_snapshot(tmp_path, rho_t=0.0)
    gate = evaluate_signal_eligibility(strict_snapshot, paper_only=True, broker_submission_allowed=False)
    target = generate_target_weights(strict_snapshot, gate, policy)
    delta = build_portfolio_delta(portfolio, target, gate, policy)

    result = simulate_hypothetical_orders(portfolio, delta, gate, policy)

    assert result.cash_after_orders >= 0.0


def test_friction_buffer_reduces_buy_notional(tmp_path: Path) -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    portfolio_path = tmp_path / "snapshot.csv"
    portfolio_path.write_text(_snapshot_csv(), encoding="utf-8")
    portfolio = load_portfolio_snapshot(portfolio_path, policy)
    strict_snapshot = _strict_snapshot(tmp_path, rho_t=0.0)
    gate = evaluate_signal_eligibility(strict_snapshot, paper_only=True, broker_submission_allowed=False)
    target = generate_target_weights(strict_snapshot, gate, policy)
    delta = build_portfolio_delta(portfolio, target, gate, policy)

    result = simulate_hypothetical_orders(portfolio, delta, gate, policy)
    buy_order = next(order for order in result.orders if order.side == "BUY")

    assert buy_order.notional < 400.0


def test_sell_orders_are_sorted_before_buy_orders(tmp_path: Path) -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    portfolio_path = tmp_path / "snapshot.csv"
    portfolio_path.write_text(_snapshot_csv(), encoding="utf-8")
    portfolio = load_portfolio_snapshot(portfolio_path, policy)
    strict_snapshot = _strict_snapshot(tmp_path, rho_t=0.0)
    gate = evaluate_signal_eligibility(strict_snapshot, paper_only=True, broker_submission_allowed=False)
    target = generate_target_weights(strict_snapshot, gate, policy)
    delta = build_portfolio_delta(portfolio, target, gate, policy)

    result = simulate_hypothetical_orders(portfolio, delta, gate, policy)

    assert [order.side for order in result.orders] == ["SELL", "BUY"]


def test_min_notional_suppresses_small_orders(tmp_path: Path) -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    portfolio_path = tmp_path / "snapshot.csv"
    portfolio_path.write_text(
        _snapshot_csv().replace("600.0,0.6", "610.0,0.61").replace("400.0,0.4", "390.0,0.39"),
        encoding="utf-8",
    )
    portfolio = load_portfolio_snapshot(portfolio_path, policy)
    strict_snapshot = _strict_snapshot(tmp_path, rho_t=0.35)
    gate = evaluate_signal_eligibility(strict_snapshot, paper_only=True, broker_submission_allowed=False)
    target = generate_target_weights(strict_snapshot, gate, policy)
    delta = build_portfolio_delta(portfolio, target, gate, policy)

    result = simulate_hypothetical_orders(portfolio, delta, gate, policy)

    assert result.orders_count == 0


def test_synthetic_strict_snapshot_uses_tmp_path_only(tmp_path: Path) -> None:
    strict_path = tmp_path / "phase14_snapshot.json"
    before_mtime = REAL_PHASE14_PATH.stat().st_mtime_ns
    _strict_snapshot(tmp_path, rho_t=0.0)

    assert str(strict_path).startswith(str(tmp_path))
    assert REAL_PHASE14_PATH.stat().st_mtime_ns == before_mtime
