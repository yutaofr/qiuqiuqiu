"""Phase 15 signal eligibility gating.

Inputs:
    A published Phase 14 cycle snapshot plus paper-only execution invariants.
Outputs:
    Deterministic eligibility and execution gating fields for downstream
    sandbox portfolio logic.
Time semantics:
    Uses only the supplied weekly snapshot, assumed known at the decision time.
As-of semantics:
    No inferred recovery or execution permissions are added after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SignalEligibilityResult:
    week_end: str
    signal_eligible: bool
    execution_allowed: bool
    reason: str
    mode: str
    backfill_mode: str | None
    strict_gate_passed: bool
    execution_permitted: bool
    h_t_available: bool
    rho_t_available: bool
    k_hat_t_available: bool
    s_t_available: bool
    paper_only: bool
    broker_submission_allowed: bool


def _value_available(value: Any) -> bool:
    return value is not None


def evaluate_signal_eligibility(
    snapshot: Mapping[str, Any],
    *,
    paper_only: bool,
    broker_submission_allowed: bool,
) -> SignalEligibilityResult:
    week_end = str(snapshot.get("week_end", ""))
    mode = str(snapshot.get("mode", ""))
    backfill_mode_raw = snapshot.get("backfill_mode")
    backfill_mode = None if backfill_mode_raw is None else str(backfill_mode_raw)
    strict_gate_passed = bool(snapshot.get("strict_gate_passed", False))
    execution_permitted = bool(snapshot.get("execution_permitted", False))
    h_t_available = _value_available(snapshot.get("h_t"))
    rho_t_available = _value_available(snapshot.get("rho_t"))
    k_hat_t_available = _value_available(snapshot.get("k_hat_t"))
    s_t_available = _value_available(snapshot.get("s_t"))

    def build(reason: str, *, signal_eligible: bool, execution_allowed: bool) -> SignalEligibilityResult:
        return SignalEligibilityResult(
            week_end=week_end,
            signal_eligible=signal_eligible,
            execution_allowed=execution_allowed,
            reason=reason,
            mode=mode,
            backfill_mode=backfill_mode,
            strict_gate_passed=strict_gate_passed,
            execution_permitted=execution_permitted,
            h_t_available=h_t_available,
            rho_t_available=rho_t_available,
            k_hat_t_available=k_hat_t_available,
            s_t_available=s_t_available,
            paper_only=paper_only,
            broker_submission_allowed=broker_submission_allowed,
        )

    if not paper_only or broker_submission_allowed:
        return build("paper_only_invariant_failed", signal_eligible=False, execution_allowed=False)
    if backfill_mode == "degraded_backfill":
        return build("degraded_backfill_signal", signal_eligible=False, execution_allowed=False)
    if backfill_mode == "block":
        return build("block_signal", signal_eligible=False, execution_allowed=False)
    if mode != "strict":
        return build("not_strict_mode", signal_eligible=False, execution_allowed=False)
    if not execution_permitted:
        return build("execution_not_permitted", signal_eligible=False, execution_allowed=False)
    if not strict_gate_passed:
        return build("strict_gate_failed", signal_eligible=False, execution_allowed=False)
    if not rho_t_available:
        return build("rho_t_missing", signal_eligible=False, execution_allowed=False)
    if not k_hat_t_available:
        return build("k_hat_t_missing", signal_eligible=False, execution_allowed=False)
    if not s_t_available:
        return build("s_t_missing", signal_eligible=False, execution_allowed=False)
    if backfill_mode not in {None, "strict_recovery"}:
        return build("not_strict_mode", signal_eligible=False, execution_allowed=False)
    return build("eligible_strict_signal", signal_eligible=True, execution_allowed=True)
