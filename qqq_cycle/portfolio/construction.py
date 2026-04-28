"""Phase 11 portfolio construction rules.

All functions consume already-produced weekly signal rows. They do not infer
future market data and they preserve the signal date as the decision timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration frozen by the Phase 11 preregistration manifest.

    Inputs:
        transaction_cost_bps: One-way cost per unit turnover, in basis points.
        turnover_threshold: Minimum QQQ weight delta required to rebalance.
        signal_time: Decision timestamp semantics for weekly signal rows.
        execution_time: Execution timestamp semantics for the backtest engine.
        allow_leverage: Must remain false for Phase 11.
        circuit_breaker_s1_index: Frozen S1 cluster integer from the manifest.
        circuit_breaker_release_weeks: Consecutive non-S1 weeks required to exit.

    Output/as-of semantics:
        The config is immutable and is applied only to rows already known at
        each weekly decision timestamp.
    """

    transaction_cost_bps: float = 5.0
    turnover_threshold: float = 0.05
    signal_time: str = "friday_close"
    execution_time: str = "next_open"
    allow_leverage: bool = False
    circuit_breaker_s1_index: int = -1
    circuit_breaker_release_weeks: int = 2


@dataclass(frozen=True)
class PortfolioWeights:
    """Auditable weekly portfolio construction output.

    Inputs:
        week_end: Friday signal date.
        rho_t, k_hat_t, drift_flag: Strict pipeline signal values known as of
            week_end.

    Outputs:
        omega_*_target are the raw rho-mapped targets when rho_t is available.
        omega_*_final are the tradable weights after circuit-breaker and
        turnover-threshold rules. Final weights are long-only and sum to one.

    Time/as-of semantics:
        The row represents a decision made at week_end for next-open execution.
    """

    week_end: str
    rho_t: float | None
    k_hat_t: int | None
    drift_flag: int
    omega_qqq_target: float | None
    omega_shy_target: float | None
    omega_qqq_final: float
    omega_shy_final: float
    rebalance_required: bool
    circuit_breaker_active: bool
    reason: str


def _clip_unit(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value)


def map_rho_to_target_weights(rho_t: float) -> tuple[float, float]:
    """Map risk score to long-only QQQ/SHY target weights.

    Input:
        rho_t: Operation risk score known at the weekly decision timestamp.

    Output:
        (omega_qqq, omega_shy), clipped to [0, 1] and normalized to sum to one.

    Time/as-of semantics:
        This pure mapping uses only rho_t from the current signal row.
    """

    if _is_missing(rho_t):
        raise ValueError("rho_t is required for target weight mapping")
    rho = _clip_unit(float(rho_t))
    omega_qqq = _clip_unit(1.0 - rho)
    omega_shy = 1.0 - omega_qqq
    if not np.isclose(omega_qqq + omega_shy, 1.0, atol=1e-12):
        raise RuntimeError("portfolio target weights do not sum to one")
    return omega_qqq, omega_shy


def apply_circuit_breaker(
    k_hat_t: int | None,
    drift_flag: int,
    breaker_active: bool,
    weeks_outside_s1: int,
    s1_index: int,
    release_weeks: int = 2,
) -> tuple[bool, int]:
    """Update circuit-breaker state using current-week signal information.

    Input:
        k_hat_t: Current state cluster, known at week_end.
        drift_flag: Current drift flag, known at week_end.
        breaker_active: Previous breaker state.
        weeks_outside_s1: Previous consecutive non-S1 counter.
        s1_index: Frozen S1 cluster integer from preregistration.
        release_weeks: Number of consecutive non-S1 weeks required to release.

    Output:
        (new_breaker_active, new_weeks_outside_s1).

    Time/as-of semantics:
        The update reads only the current row and prior breaker state.
    """

    if s1_index < 0:
        return False, 0

    cluster = None if _is_missing(k_hat_t) else int(k_hat_t)
    triggered = cluster == int(s1_index) and int(drift_flag) == 1
    if triggered:
        return True, 0

    if not breaker_active:
        return False, 0

    if cluster is not None and cluster != int(s1_index):
        next_outside = int(weeks_outside_s1) + 1
        if next_outside >= int(release_weeks):
            return False, next_outside
        return True, next_outside

    return True, 0


def apply_turnover_threshold(
    prev_omega_qqq: float,
    target_omega_qqq: float,
    threshold: float = 0.05,
) -> tuple[float, bool]:
    """Apply the Phase 11 rebalance threshold to a QQQ target weight.

    Input:
        prev_omega_qqq: Previous final QQQ weight.
        target_omega_qqq: Current desired QQQ target weight.
        threshold: Minimum absolute delta required to rebalance.

    Output:
        (final_omega_qqq, rebalance_required).

    Time/as-of semantics:
        Uses only the prior final weight and current target weight.
    """

    prev = _clip_unit(float(prev_omega_qqq))
    target = _clip_unit(float(target_omega_qqq))
    if abs(target - prev) < float(threshold):
        return prev, False
    return target, True


def _extract_interpretability(raw: object) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        return {}
    return parsed


def compute_s1_cluster_index(pipeline_df: pd.DataFrame) -> int:
    """Derive the semantic S1 cluster integer from strict pipeline output.

    Input:
        pipeline_df: Pipeline output with k_hat_t and interpretability JSON
            containing H and I values.

    Output:
        The cluster integer whose median-H/median-I partition maps to S1.

    Time/as-of semantics:
        Intended for preregistration before backtest execution. It summarizes
        the already-frozen Phase 10 strict output and must be written to the
        manifest before returns are computed.
    """

    records: list[dict[str, float | int]] = []
    for _, row in pipeline_df.iterrows():
        if _is_missing(row.get("k_hat_t")):
            continue
        interp = _extract_interpretability(row.get("interpretability"))
        h_val = interp.get("H")
        i_val = interp.get("I", row.get("I_t"))
        if _is_missing(h_val) or _is_missing(i_val):
            continue
        records.append(
            {
                "k_hat_t": int(row["k_hat_t"]),
                "H": float(h_val),
                "I": float(i_val),
            }
        )

    medians = pd.DataFrame.from_records(records).groupby("k_hat_t")[["H", "I"]].median()
    if len(medians) < 5:
        raise ValueError("need all five clusters with finite H/I to derive S1")

    low_h_clusters = medians["H"].sort_values().index[:2]
    return int(medians.loc[low_h_clusters, "I"].idxmin())


def build_weekly_weights(
    signal_df: pd.DataFrame,
    config: BacktestConfig,
) -> list[PortfolioWeights]:
    """Build weekly final weights from strict signal rows.

    Input:
        signal_df: DataFrame with week_end, rho_t, k_hat_t, drift_flag columns.
        config: Frozen Phase 11 construction config.

    Output:
        One PortfolioWeights row per input signal row.

    Time/as-of semantics:
        Iterates in week_end order. Row t uses only row t signal values and
        previous portfolio state; it does not inspect future signals or prices.
    """

    required = {"week_end", "rho_t", "k_hat_t", "drift_flag"}
    missing = required.difference(signal_df.columns)
    if missing:
        raise ValueError(f"signal_df missing required columns: {sorted(missing)}")
    if config.allow_leverage:
        raise ValueError("Phase 11 forbids leverage")

    rows: list[PortfolioWeights] = []
    prev_omega_qqq = 0.5
    breaker_active = False
    weeks_outside_s1 = 0

    ordered = signal_df.copy()
    ordered["week_end"] = pd.to_datetime(ordered["week_end"])
    ordered = ordered.sort_values("week_end", kind="mergesort")

    for _, row in ordered.iterrows():
        rho_raw = row["rho_t"]
        k_hat_raw = row["k_hat_t"]
        drift_flag = 0 if _is_missing(row["drift_flag"]) else int(row["drift_flag"])
        k_hat = None if _is_missing(k_hat_raw) else int(k_hat_raw)

        target_qqq: float | None
        target_shy: float | None
        reason: str
        if _is_missing(rho_raw):
            target_qqq = None
            target_shy = None
            final_qqq = prev_omega_qqq
            rebalance_required = False
            reason = "rho_t_missing"
        else:
            rho_t = float(rho_raw)
            target_qqq, target_shy = map_rho_to_target_weights(rho_t)
            final_qqq, rebalance_required = apply_turnover_threshold(
                prev_omega_qqq,
                target_qqq,
                config.turnover_threshold,
            )
            reason = "rebalance" if rebalance_required else "turnover_below_threshold"

        breaker_active, weeks_outside_s1 = apply_circuit_breaker(
            k_hat_t=k_hat,
            drift_flag=drift_flag,
            breaker_active=breaker_active,
            weeks_outside_s1=weeks_outside_s1,
            s1_index=config.circuit_breaker_s1_index,
            release_weeks=config.circuit_breaker_release_weeks,
        )
        if breaker_active:
            final_qqq = 0.0
            rebalance_required = not np.isclose(prev_omega_qqq, 0.0, atol=1e-12)
            reason = "circuit_breaker"

        final_qqq = _clip_unit(final_qqq)
        final_shy = 1.0 - final_qqq
        if not np.isclose(final_qqq + final_shy, 1.0, atol=1e-12):
            raise RuntimeError("final portfolio weights do not sum to one")
        if final_qqq < 0.0 or final_shy < 0.0:
            raise RuntimeError("negative portfolio weight produced")

        rows.append(
            PortfolioWeights(
                week_end=row["week_end"].strftime("%Y-%m-%d"),
                rho_t=None if _is_missing(rho_raw) else float(rho_raw),
                k_hat_t=k_hat,
                drift_flag=drift_flag,
                omega_qqq_target=target_qqq,
                omega_shy_target=target_shy,
                omega_qqq_final=final_qqq,
                omega_shy_final=final_shy,
                rebalance_required=bool(rebalance_required),
                circuit_breaker_active=bool(breaker_active),
                reason=reason,
            )
        )
        prev_omega_qqq = final_qqq

    return rows
