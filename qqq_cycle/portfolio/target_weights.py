"""Phase 15 target weight generation.

Inputs:
    Current cycle snapshot, signal gate result, frozen portfolio policy, and
    an optional prior target.
Outputs:
    Long-only target weights suitable for paper-only sandbox execution.
Time semantics:
    Uses only the current week's snapshot, current policy, and prior target.
As-of semantics:
    No future prices, fills, or smoothing adjustments are introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from qqq_cycle.portfolio.policy import PortfolioPolicy
from qqq_cycle.portfolio.signal_gate import SignalEligibilityResult


EPSILON = 1e-9


@dataclass(frozen=True)
class TargetWeightsResult:
    week_end: str
    target_weights: dict[str, float]
    policy_id: str
    policy_version: int
    signal_eligible: bool
    generation_mode: str
    rho_bucket: str | None
    reason: str
    gross_exposure: float
    net_exposure: float
    paper_only: bool
    broker_submission_allowed: bool
    known_limitations: list[str]


def _weight_sums_valid(weights: Mapping[str, float]) -> bool:
    return abs(sum(weights.values()) - 1.0) <= EPSILON


def _validate_weight_symbols(weights: Mapping[str, float], policy: PortfolioPolicy) -> dict[str, float]:
    normalized = {str(symbol): float(value) for symbol, value in weights.items()}
    unknown = set(normalized) - set(policy.symbols)
    if unknown:
        raise ValueError(f"target weights reference unknown assets: {sorted(unknown)}")
    if not _weight_sums_valid(normalized):
        raise ValueError("target weights must sum to 1.0")
    return normalized


def _defensive_cash_proxy_weights(policy: PortfolioPolicy) -> dict[str, float]:
    weights = {symbol: 0.0 for symbol in policy.symbols}
    cash_proxies = [asset.symbol for asset in policy.universe if asset.asset_class == "cash_proxy"]
    if not cash_proxies:
        raise ValueError("portfolio policy must define a cash_proxy asset for defensive fallback")
    weights[cash_proxies[0]] = 1.0
    return weights


def _compute_exposures(weights: Mapping[str, float]) -> tuple[float, float]:
    gross = float(sum(abs(value) for value in weights.values()))
    net = float(sum(weights.values()))
    return gross, net


def generate_target_weights(
    snapshot: Mapping[str, Any],
    signal_gate: SignalEligibilityResult,
    policy: PortfolioPolicy,
    prior_target_weights: Mapping[str, float] | None = None,
) -> TargetWeightsResult:
    if signal_gate.paper_only is not True or signal_gate.broker_submission_allowed is not False:
        raise ValueError("target weights require paper_only=true and broker_submission_allowed=false")

    limitations = list(policy.known_limitations)
    week_end = str(snapshot.get("week_end", signal_gate.week_end))
    if not signal_gate.signal_eligible:
        if prior_target_weights is not None:
            target_weights = _validate_weight_symbols(prior_target_weights, policy)
            reason = signal_gate.reason
        else:
            target_weights = _defensive_cash_proxy_weights(policy)
            reason = "no_prior_target_defensive_cash_proxy"
        gross_exposure, net_exposure = _compute_exposures(target_weights)
        return TargetWeightsResult(
            week_end=week_end,
            target_weights=target_weights,
            policy_id=policy.policy_id,
            policy_version=policy.policy_version,
            signal_eligible=False,
            generation_mode="hold_prior_or_policy_default",
            rho_bucket=None,
            reason=reason,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            paper_only=True,
            broker_submission_allowed=False,
            known_limitations=limitations,
        )

    rho_t = snapshot.get("rho_t")
    if rho_t is None:
        raise ValueError("strict eligible target generation requires rho_t")
    rho_bucket = policy.locate_rho_bucket(float(rho_t))
    target_weights = _validate_weight_symbols(policy.default_state_policy()[rho_bucket], policy)
    gross_exposure, net_exposure = _compute_exposures(target_weights)
    return TargetWeightsResult(
        week_end=week_end,
        target_weights=target_weights,
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
        signal_eligible=True,
        generation_mode="policy_bucket_mapping",
        rho_bucket=rho_bucket,
        reason="eligible_strict_signal",
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        paper_only=True,
        broker_submission_allowed=False,
        known_limitations=limitations,
    )
