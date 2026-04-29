#!/usr/bin/env python3
"""Run the Phase 15 paper-only execution sandbox."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qqq_cycle.portfolio.delta import build_portfolio_delta
from qqq_cycle.portfolio.order_simulator import simulate_hypothetical_orders
from qqq_cycle.portfolio.policy import load_portfolio_policy
from qqq_cycle.portfolio.portfolio_snapshot import load_portfolio_snapshot
from qqq_cycle.portfolio.reporting import write_phase15_artifacts
from qqq_cycle.portfolio.signal_gate import evaluate_signal_eligibility
from qqq_cycle.portfolio.target_weights import generate_target_weights


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_phase14_snapshot(week_end: str, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    latest = Path("outputs/phase14/cycle_snapshot_latest.json")
    if latest.exists():
        payload = _load_json(latest)
        if str(payload.get("week_end")) == week_end:
            return latest
    history_dir = Path("outputs/phase14/history")
    matches = sorted(history_dir.glob(f"cycle_snapshot_{week_end}__run_*.json"))
    if not matches:
        raise FileNotFoundError(f"no Phase 14 snapshot found for week_end={week_end}")
    return matches[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week-end", required=True)
    parser.add_argument("--phase14-snapshot", type=Path, default=None)
    parser.add_argument("--policy", type=Path, default=Path("configs/portfolio_policy_v1.yaml"))
    parser.add_argument("--portfolio-snapshot", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase15"))
    args = parser.parse_args()

    phase14_snapshot_path = _resolve_phase14_snapshot(args.week_end, args.phase14_snapshot)
    portfolio_snapshot_path = (
        args.portfolio_snapshot
        if args.portfolio_snapshot is not None
        else Path("sandbox/portfolio") / f"current_positions_{args.week_end}.csv"
    )
    phase14_snapshot = _load_json(phase14_snapshot_path)
    policy = load_portfolio_policy(args.policy)
    portfolio_snapshot = load_portfolio_snapshot(portfolio_snapshot_path, policy)
    signal_gate = evaluate_signal_eligibility(
        phase14_snapshot,
        paper_only=policy.paper_only,
        broker_submission_allowed=policy.broker_submission_allowed,
    )
    target = generate_target_weights(
        phase14_snapshot,
        signal_gate,
        policy,
        prior_target_weights=portfolio_snapshot.weights,
    )
    delta = build_portfolio_delta(portfolio_snapshot, target, signal_gate, policy)
    orders = simulate_hypothetical_orders(portfolio_snapshot, delta, signal_gate, policy)
    artifacts = write_phase15_artifacts(
        output_dir=args.output_dir,
        week_end=args.week_end,
        phase14_snapshot=phase14_snapshot,
        signal_gate=signal_gate,
        target=target,
        delta=delta,
        orders=orders,
    )
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
