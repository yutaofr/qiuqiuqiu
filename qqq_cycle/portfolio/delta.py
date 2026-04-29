"""Phase 15 portfolio delta engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from qqq_cycle.portfolio.policy import PortfolioPolicy
from qqq_cycle.portfolio.portfolio_snapshot import PaperPortfolioSnapshot
from qqq_cycle.portfolio.signal_gate import SignalEligibilityResult
from qqq_cycle.portfolio.target_weights import TargetWeightsResult


EPSILON = 1e-9


@dataclass(frozen=True)
class PortfolioDelta:
    week_end: str
    nav: float
    current_weights: dict[str, float]
    target_weights: dict[str, float]
    delta_weights: dict[str, float]
    turnover: float
    rebalance_required: bool
    reason: str
    paper_only: bool
    broker_submission_allowed: bool


def _validate_symbols(weights: Mapping[str, float], policy: PortfolioPolicy, label: str) -> dict[str, float]:
    normalized = {str(symbol): float(value) for symbol, value in weights.items()}
    unknown = set(normalized).difference(policy.symbols)
    if unknown:
        raise ValueError(f"{label} references unknown symbols: {sorted(unknown)}")
    if abs(sum(normalized.values()) - 1.0) > 0.01:
        raise ValueError(f"{label} weights must sum to approximately 1.0")
    return normalized


def build_portfolio_delta(
    snapshot: PaperPortfolioSnapshot,
    target: TargetWeightsResult,
    signal_gate: SignalEligibilityResult,
    policy: PortfolioPolicy,
) -> PortfolioDelta:
    if snapshot.paper_only is not True or target.paper_only is not True or signal_gate.paper_only is not True:
        raise ValueError("portfolio delta requires paper_only=true for all inputs")
    if (
        snapshot.broker_submission_allowed is not False
        or target.broker_submission_allowed is not False
        or signal_gate.broker_submission_allowed is not False
    ):
        raise ValueError("portfolio delta requires broker_submission_allowed=false for all inputs")

    current_weights = _validate_symbols(snapshot.weights, policy, "current portfolio")
    target_weights = _validate_symbols(target.target_weights, policy, "target portfolio")
    delta_weights = {
        symbol: float(target_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0))
        for symbol in policy.symbols
    }
    turnover = 0.5 * sum(abs(value) for value in delta_weights.values())
    if not signal_gate.execution_allowed:
        rebalance_required = False
        reason = signal_gate.reason
    elif turnover < policy.constraints.turnover_threshold - EPSILON:
        rebalance_required = False
        reason = "turnover_below_threshold"
    else:
        rebalance_required = True
        reason = "rebalance_required"

    return PortfolioDelta(
        week_end=target.week_end,
        nav=float(snapshot.nav),
        current_weights=current_weights,
        target_weights=target_weights,
        delta_weights=delta_weights,
        turnover=float(turnover),
        rebalance_required=rebalance_required,
        reason=reason,
        paper_only=True,
        broker_submission_allowed=False,
    )
