"""Live weekly execution runtime.

Each call to LiveRuntime.run_week() consumes one new week of macro data, loads
the prior persisted state, advances the pipeline and portfolio state by one
step, and writes the updated state and run artifacts to disk.

The "single-week incremental" guarantee: covariance, prototype, IIR envelope,
and circuit-breaker state are loaded from persistence — never recomputed from
full history. The batch layer computations (state_layer, stress_layer,
drift_probe) run on a bounded macro_tail (last ~600 rows) which is stored in
the state directory and updated each run.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qqq_cycle.backtest.diagnostics import _semantic_label, _state_probabilities
from qqq_cycle.config import ModelConfig, load_config
from qqq_cycle.core.covariance import CovarianceState2D, RobustEWCov2D
from qqq_cycle.core.drift_probe import DriftProbe
from qqq_cycle.core.proto_online import (
    PrototypeState,
    initialize_prototypes_from_history,
    update_prototypes,
)
from qqq_cycle.core.risk_layer import RiskScore, blended_state_weight, compute_risk_score
from qqq_cycle.core.state_layer import compute_state_layer
from qqq_cycle.core.stress_layer import compute_stress_layer
from qqq_cycle.pipeline import (
    PipelineContracts,
    PipelineResult,
    _check_strict_gate,
    _safe_float,
    _build_audit_interpretability,
    _build_interpretability,
    _compute_weekly_h_t_from_stores,
    MODE_WARMUP,
    MODE_DEGRADED,
    MODE_STRICT,
    _HEAL_CIRCUIT_WEEKS,
)
from qqq_cycle.portfolio.construction import (
    BacktestConfig,
    PortfolioWeights,
    apply_circuit_breaker,
    apply_turnover_threshold,
    map_rho_to_target_weights,
    _is_missing,
)
from qqq_cycle.live.dashboard import (
    append_drift_monitor,
    append_pollution_flags,
    append_state_plane,
)
from qqq_cycle.live.freshness import (
    FreshnessRecord,
    check_all_freshness,
    derive_execution_state,
)
from qqq_cycle.live.interpretability import InterpretabilitySnapshot, build_snapshot, snapshot_to_dict
from qqq_cycle.live.state_io import LiveState, StateNotAvailableError, load_state, save_state

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveRunResult:
    """Complete output of one weekly live run."""

    asof_week_end: str
    mode: str                             # warmup / degraded / strict (pipeline mode)
    execution_state: str                  # execute / degrade / block
    signal_bundle: dict[str, Any]
    portfolio_bundle: dict[str, Any]
    interpretability_bundle: dict[str, Any]
    state_path: str
    degraded_reason: str | None
    execution_permitted: bool
    execution_block_reason: str | None
    signal_valid_but_not_executable: bool
    freshness_snapshot: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Incremental pipeline step
# ---------------------------------------------------------------------------

def _run_pipeline_step(
    *,
    week_end: pd.Timestamp,
    theta_row: np.ndarray,
    s_t_val: float | None,
    i_t_val: float | None,
    drift_raw: float | None,
    drift_flag: int,
    h_t_raw: float | None,
    cov_state: CovarianceState2D,
    proto: PrototypeState | None,
    proto_seed: list[np.ndarray],
    h_t_lead_prev: float,
    heal_count: int,
    config: ModelConfig,
    can_compute_h_t: bool,
    degraded_reason: str | None,
    week_index: int,
    # Pass precomputed state/stress/drift frames for interpretability builder
    state_frame: pd.DataFrame,
    stress_frame: pd.DataFrame,
    drift_frame: pd.DataFrame,
) -> tuple[
    PipelineResult,
    CovarianceState2D,
    PrototypeState | None,
    list[np.ndarray],
    float,
    int,
]:
    """Advance the pipeline by one week and return the updated state.

    This is the loop body from run_pipeline() lifted into a standalone function
    so the live kernel can load prior state instead of replaying all history.

    Returns:
        (result, new_cov_state, new_proto, new_proto_seed, new_h_t_lead_prev, new_heal_count)
    """
    cov = RobustEWCov2D(warmup_weeks=config.warmup_weeks)
    x = theta_row

    if not cov.is_warm(cov_state):
        # Warmup mode — advance state but suppress outputs.
        if np.all(np.isfinite(x)):
            cov_state = cov.update(cov_state, x)
            proto_seed = proto_seed + [x]
        else:
            cov_state = cov.update(cov_state, np.array([np.nan, np.nan]))
        result = PipelineResult(
            week_end=week_end.strftime("%Y-%m-%d"),
            k_hat_t=None, p_t=None, s_t=None, h_t=None,
            rho_t=None, I_t=None, interpretability=None,
            mode=MODE_WARMUP, degraded_reason=None,
            strict_contracts_satisfied=None,
        )
        return result, cov_state, proto, proto_seed, h_t_lead_prev, heal_count

    # Post-warmup: update covariance + prototype.
    k_hat_t: int | None = None
    p_t: list[float] | None = None

    if np.all(np.isfinite(x)):
        if proto is None and len(proto_seed) >= config.warmup_weeks:
            proto = initialize_prototypes_from_history(np.asarray(proto_seed))
            proto_seed = []
        if proto is not None:
            prev_cov = cov_state.cov_reg.copy()
            cov_state = cov.update(cov_state, x)
            proto_result = update_prototypes(
                proto, x, cov_state.mean, prev_cov, cov_state.cov_reg, week_index
            )
            proto = proto_result.state
            probs = _state_probabilities(x, proto, cov_state.cov_reg)
            k_hat_t = int(np.argmax(probs))
            p_t = [float(p) for p in probs]
        else:
            cov_state = cov.update(cov_state, x)
    else:
        cov_state = cov.update(cov_state, np.array([np.nan, np.nan]))

    s_t = s_t_val

    # Strict gate: h_t and rho_t.
    h_t: float | None = None
    rho_t: float | None = None
    strict_contracts_satisfied: bool | None = False
    omega_t: float | None = None
    n_t: float | None = None
    omega_state = np.asarray(config.risk.omega_state, dtype=float)

    if can_compute_h_t and h_t_raw is not None:
        delta_abs = abs(drift_raw) if drift_raw is not None else 0.0
        h_t_lead = max(h_t_raw, config.micro.iir_delta * h_t_lead_prev)
        if h_t_raw < config.micro.heal_threshold:
            heal_count += 1
            if heal_count >= _HEAL_CIRCUIT_WEEKS:
                h_t_lead = h_t_raw
                heal_count = 0
        else:
            heal_count = 0
        h_t_lead_prev = h_t_lead
        h_t = h_t_raw
        if p_t is not None and s_t is not None:
            omega_t = blended_state_weight(
                np.asarray(p_t, dtype=float),
                omega_state,
                delta_abs,
                theta_lo=config.drift.theta_lo,
                theta_hi=config.drift.theta_hi,
            )
            risk: RiskScore = compute_risk_score(
                omega_t=omega_t,
                s_t=s_t,
                h_t_lead=h_t_lead,
                lambda_rho=config.risk.lambda_rho,
            )
            n_t = float(risk.n_t)
            rho_t = float(risk.rho_t)
        strict_contracts_satisfied = True

    mode = MODE_STRICT if h_t is not None else MODE_DEGRADED
    if mode == MODE_STRICT:
        row_degraded_reason = None
    elif can_compute_h_t and h_t is None:
        row_degraded_reason = "h_t unavailable for this week: micro data window not satisfied"
    else:
        row_degraded_reason = degraded_reason

    interp = _build_interpretability(
        week_end, state_frame, stress_frame, drift_frame, k_hat_t, p_t, h_t, rho_t
    )
    I_t = _build_audit_interpretability(
        week_end,
        state_frame,
        stress_frame,
        drift_frame,
        omega_t=omega_t,
        s_t=s_t,
        n_t=n_t,
        h_t=h_t,
        h_t_available=h_t is not None,
        rho_t_available=rho_t is not None,
        config=config,
    )

    result = PipelineResult(
        week_end=week_end.strftime("%Y-%m-%d"),
        k_hat_t=k_hat_t,
        p_t=p_t,
        s_t=s_t,
        h_t=h_t,
        rho_t=rho_t,
        I_t=I_t,
        interpretability=interp,
        mode=mode,
        degraded_reason=row_degraded_reason,
        strict_contracts_satisfied=bool(strict_contracts_satisfied) if strict_contracts_satisfied is not None else None,
    )
    return result, cov_state, proto, proto_seed, h_t_lead_prev, heal_count


def _run_portfolio_step(
    result: PipelineResult,
    breaker_active: bool,
    weeks_outside_s1: int,
    prev_omega_qqq: float,
    backtest_config: BacktestConfig,
) -> tuple[PortfolioWeights, bool, int]:
    """Advance portfolio construction by one week and return updated CB state."""
    rho_raw = result.rho_t
    k_hat = result.k_hat_t
    drift_flag = 0

    if result.interpretability:
        drift_flag = int(result.interpretability.get("drift_flag", 0))

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
        rho_t = float(rho_raw)  # type: ignore[arg-type]
        target_qqq, target_shy = map_rho_to_target_weights(rho_t)
        final_qqq, rebalance_required = apply_turnover_threshold(
            prev_omega_qqq, target_qqq, backtest_config.turnover_threshold
        )
        reason = "rebalance" if rebalance_required else "turnover_below_threshold"

    breaker_active, weeks_outside_s1 = apply_circuit_breaker(
        k_hat_t=k_hat,
        drift_flag=drift_flag,
        breaker_active=breaker_active,
        weeks_outside_s1=weeks_outside_s1,
        s1_index=backtest_config.circuit_breaker_s1_index,
        release_weeks=backtest_config.circuit_breaker_release_weeks,
    )
    if breaker_active:
        final_qqq = 0.0
        rebalance_required = not np.isclose(prev_omega_qqq, 0.0, atol=1e-12)
        reason = "circuit_breaker"

    final_qqq = float(min(1.0, max(0.0, final_qqq)))
    final_shy = 1.0 - final_qqq

    weights = PortfolioWeights(
        week_end=result.week_end,
        rho_t=None if _is_missing(rho_raw) else float(rho_raw),  # type: ignore[arg-type]
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
    return weights, breaker_active, weeks_outside_s1


# ---------------------------------------------------------------------------
# Live runtime
# ---------------------------------------------------------------------------

class LiveRuntime:
    """Weekly incremental live execution engine."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        backtest_config: BacktestConfig | None = None,
    ) -> None:
        self.config = config or load_config()
        self.backtest_config = backtest_config or BacktestConfig()

    def run_week(
        self,
        *,
        week_end: str,
        macro_row: pd.Series,
        contracts: PipelineContracts | None = None,
        state_dir: Path,
        output_dir: Path,
    ) -> LiveRunResult:
        """Run one weekly live cycle.

        Args:
            week_end: ISO date string of the Friday decision date.
            macro_row: Single-row Series with macro columns for this week.
                Must include: DFII10, DGS2, BAMLH0A0HYM2, NFCI, VIXCLS,
                AI_GPR, USEPUINDXD, QQQ.
            contracts: Optional strict input contracts for h_t computation.
            state_dir: Directory holding live_state_latest/.
            output_dir: Directory for live_run_log.csv and live_run_summary.json.

        Returns:
            LiveRunResult with signal, portfolio, and execution state.

        Raises:
            StateNotAvailableError: If state cannot be loaded.
        """
        week_ts = pd.Timestamp(week_end)
        run_ts = datetime.now(timezone.utc).isoformat()

        # --- 1. Load state (fail-closed) ---
        state = load_state(state_dir)

        # --- 2. Extend macro tail ---
        new_row_df = pd.DataFrame([macro_row.to_dict()], index=[week_ts])
        macro_tail = pd.concat([state.macro_tail, new_row_df])
        macro_tail = macro_tail[~macro_tail.index.duplicated(keep="last")]
        macro_tail = macro_tail.sort_index()

        # --- 3. Freshness checks ---
        freshness = check_all_freshness(
            macro_df=macro_tail,
            week_end=week_ts,
            constituent_store=contracts.constituent_store if contracts else None,
            weight_store=contracts.weight_store if contracts else None,
            pit_engine=contracts.pit_engine if contracts else None,
        )

        # --- 4. Resolve h_t_raw from stores if available ---
        resolved_contracts = contracts
        if (
            contracts is not None
            and contracts.pit_engine is not None
            and contracts.constituent_store is not None
            and contracts.weight_store is not None
        ):
            single_week_index = pd.DatetimeIndex([week_ts])
            computed_h_t = _compute_weekly_h_t_from_stores(
                single_week_index, contracts, self.config
            )
            resolved_contracts = PipelineContracts(
                weekly_h_t=computed_h_t,
                pit_engine=contracts.pit_engine,
                constituent_store=contracts.constituent_store,
                weight_store=contracts.weight_store,
                pit_engine_available=True,
                constituents_available=True,
                weights_available=True,
            )

        can_compute_h_t, base_degraded_reason = _check_strict_gate(resolved_contracts)

        # --- 5. Batch layer computations on the macro tail ---
        state_frame = compute_state_layer(macro_tail)
        theta = state_frame[["H", "I"]]
        stress_result = compute_stress_layer(theta, state_frame["E"])
        stress_frame = stress_result.frame
        drift_frame = DriftProbe(
            theta_lo=self.config.drift.theta_lo,
            theta_hi=self.config.drift.theta_hi,
        ).compute(macro_tail)

        # Extract current week values from batch outputs
        theta_row_vals = (
            theta.loc[week_ts].to_numpy(dtype=float)
            if week_ts in theta.index
            else np.array([np.nan, np.nan])
        )
        s_t_val = (
            _safe_float(stress_frame.at[week_ts, "s"])
            if week_ts in stress_frame.index else None
        )
        i_t_val = (
            float(theta_row_vals[1]) if np.isfinite(theta_row_vals[1]) else None
        )
        drift_raw = (
            _safe_float(drift_frame.at[week_ts, "drift_probe_raw"])
            if week_ts in drift_frame.index else None
        )
        drift_flag = (
            int(drift_frame.at[week_ts, "drift_flag"])
            if week_ts in drift_frame.index
            and pd.notna(drift_frame.at[week_ts, "drift_flag"])
            else 0
        )

        h_t_raw: float | None = None
        if can_compute_h_t and resolved_contracts is not None:
            h_t_series = resolved_contracts.weekly_h_t
            if h_t_series is not None and week_ts in h_t_series.index:
                h_t_raw = _safe_float(h_t_series.loc[week_ts])

        # --- 6. Incremental pipeline step ---
        (
            pipeline_result,
            new_cov,
            new_proto,
            new_proto_seed,
            new_h_t_lead_prev,
            new_heal_count,
        ) = _run_pipeline_step(
            week_end=week_ts,
            theta_row=theta_row_vals,
            s_t_val=s_t_val,
            i_t_val=i_t_val,
            drift_raw=drift_raw,
            drift_flag=drift_flag,
            h_t_raw=h_t_raw,
            cov_state=state.cov_state,
            proto=state.proto,
            proto_seed=state.proto_seed,
            h_t_lead_prev=state.h_t_lead_prev,
            heal_count=state.heal_count,
            config=self.config,
            can_compute_h_t=can_compute_h_t,
            degraded_reason=base_degraded_reason,
            week_index=state.warmup_count,
            state_frame=state_frame,
            stress_frame=stress_frame,
            drift_frame=drift_frame,
        )

        # --- 7. Portfolio step ---
        portfolio_weights, new_breaker, new_weeks_outside = _run_portfolio_step(
            pipeline_result,
            state.breaker_active,
            state.weeks_outside_s1,
            state.prev_omega_qqq,
            self.backtest_config,
        )

        # --- 8. Determine execution state ---
        pipeline_mode = pipeline_result.mode
        execution_state, block_reason = derive_execution_state(freshness, pipeline_mode)
        execution_permitted = execution_state == "execute"
        signal_valid_but_not_executable = (
            execution_state == "degrade" and pipeline_mode == MODE_STRICT
        )

        # --- 9. Persist updated state ---
        new_state = LiveState(
            week_end=week_end,
            cov_state=new_cov,
            proto=new_proto,
            proto_seed=new_proto_seed,
            h_t_lead_prev=new_h_t_lead_prev,
            heal_count=new_heal_count,
            warmup_count=new_cov.warmup_count,
            breaker_active=new_breaker,
            weeks_outside_s1=new_weeks_outside,
            prev_omega_qqq=portfolio_weights.omega_qqq_final,
            macro_tail=macro_tail,
            last_successful_timestamps={
                **state.last_successful_timestamps,
                "last_run": run_ts,
                "last_week_end": week_end,
            },
        )
        save_state(new_state, state_dir)

        # --- 10. Write run artifacts ---
        output_dir.mkdir(parents=True, exist_ok=True)
        freshness_dicts = [
            {
                "source_label": r.source_label,
                "last_observation_date": r.last_observation_date,
                "fresh_enough": r.fresh_enough,
                "blocking_level": r.blocking_level,
                "reason": r.reason,
            }
            for r in freshness
        ]
        signal_bundle = {
            "week_end": pipeline_result.week_end,
            "mode": pipeline_result.mode,
            "k_hat_t": pipeline_result.k_hat_t,
            "s_t": pipeline_result.s_t,
            "h_t": pipeline_result.h_t,
            "rho_t": pipeline_result.rho_t,
            "I_t": asdict(pipeline_result.I_t) if pipeline_result.I_t is not None else None,
        }
        portfolio_bundle = {
            "omega_qqq_target": portfolio_weights.omega_qqq_target,
            "omega_shy_target": portfolio_weights.omega_shy_target,
            "omega_qqq_final": portfolio_weights.omega_qqq_final,
            "omega_shy_final": portfolio_weights.omega_shy_final,
            "rebalance_required": portfolio_weights.rebalance_required,
            "circuit_breaker_active": portfolio_weights.circuit_breaker_active,
            "reason": portfolio_weights.reason,
        }
        interpretability_bundle = pipeline_result.interpretability or {}

        _append_run_log(output_dir, week_end, run_ts, pipeline_result, portfolio_weights,
                        execution_state, block_reason, freshness)
        _write_run_summary(output_dir, week_end, run_ts, pipeline_result, portfolio_weights,
                           execution_state, block_reason, freshness_dicts)

        # --- 11. Build + persist interpretability snapshot ---
        snap = build_snapshot(
            week_end=week_end,
            pipeline_result=pipeline_result,
            freshness=freshness,
            execution_state=execution_state,
            execution_permitted=execution_permitted,
            signal_valid_but_not_executable=signal_valid_but_not_executable,
            live_state=new_state,
            config=self.config,
        )
        (output_dir / "interpretability_snapshot_latest.json").write_text(
            json.dumps(snapshot_to_dict(snap), indent=2, default=str), encoding="utf-8"
        )
        append_state_plane(snap, output_dir)
        append_drift_monitor(snap, output_dir)
        append_pollution_flags(snap, output_dir)

        return LiveRunResult(
            asof_week_end=week_end,
            mode=pipeline_mode,
            execution_state=execution_state,
            signal_bundle=signal_bundle,
            portfolio_bundle=portfolio_bundle,
            interpretability_bundle=interpretability_bundle,
            state_path=str(state_dir / "live_state_latest"),
            degraded_reason=pipeline_result.degraded_reason,
            execution_permitted=execution_permitted,
            execution_block_reason=block_reason,
            signal_valid_but_not_executable=signal_valid_but_not_executable,
            freshness_snapshot=freshness_dicts,
        )


# ---------------------------------------------------------------------------
# Run artifact helpers
# ---------------------------------------------------------------------------

_LOG_FIELDNAMES = [
    "run_timestamp", "week_end", "mode", "execution_state", "execution_permitted",
    "degraded_reason", "block_reason", "strict_contracts_satisfied",
    "k_hat_t", "s_t", "h_t", "rho_t",
    "omega_qqq_final", "circuit_breaker_active", "reason",
    "freshness_block_count", "freshness_degrade_count",
]


def _append_run_log(
    output_dir: Path,
    week_end: str,
    run_ts: str,
    pr: PipelineResult,
    pw: PortfolioWeights,
    execution_state: str,
    block_reason: str | None,
    freshness: list[FreshnessRecord],
) -> None:
    log_path = output_dir / "live_run_log.csv"
    write_header = not log_path.exists()
    block_count = sum(1 for r in freshness if not r.fresh_enough and r.blocking_level == "block")
    degrade_count = sum(1 for r in freshness if not r.fresh_enough and r.blocking_level == "degrade")
    row = {
        "run_timestamp": run_ts,
        "week_end": week_end,
        "mode": pr.mode,
        "execution_state": execution_state,
        "execution_permitted": execution_state == "execute",
        "degraded_reason": pr.degraded_reason or "",
        "block_reason": block_reason or "",
        "strict_contracts_satisfied": pr.strict_contracts_satisfied,
        "k_hat_t": pr.k_hat_t,
        "s_t": pr.s_t,
        "h_t": pr.h_t,
        "rho_t": pr.rho_t,
        "omega_qqq_final": pw.omega_qqq_final,
        "circuit_breaker_active": pw.circuit_breaker_active,
        "reason": pw.reason,
        "freshness_block_count": block_count,
        "freshness_degrade_count": degrade_count,
    }
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_run_summary(
    output_dir: Path,
    week_end: str,
    run_ts: str,
    pr: PipelineResult,
    pw: PortfolioWeights,
    execution_state: str,
    block_reason: str | None,
    freshness_dicts: list[dict[str, Any]],
) -> None:
    summary = {
        "run_timestamp": run_ts,
        "week_end": week_end,
        "mode": pr.mode,
        "execution_state": execution_state,
        "execution_permitted": execution_state == "execute",
        "signal_valid_but_not_executable": (
            execution_state == "degrade" and pr.mode == MODE_STRICT
        ),
        "degraded_reason": pr.degraded_reason,
        "execution_block_reason": block_reason,
        "strict_contracts_satisfied": pr.strict_contracts_satisfied,
        "k_hat_t": pr.k_hat_t,
        "p_t": pr.p_t,
        "s_t": pr.s_t,
        "h_t": pr.h_t,
        "rho_t": pr.rho_t,
        "I_t": asdict(pr.I_t) if pr.I_t is not None else None,
        "interpretability": pr.interpretability,
        "omega_qqq_final": pw.omega_qqq_final,
        "omega_shy_final": pw.omega_shy_final,
        "circuit_breaker_active": pw.circuit_breaker_active,
        "rebalance_required": pw.rebalance_required,
        "reason": pw.reason,
        "freshness": freshness_dicts,
    }
    (output_dir / "live_run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
