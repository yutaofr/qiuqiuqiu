"""Unified pipeline entry point for the QQQ cycle-state model.

Provides a single auditable execution loop that routes each week to one of
three modes based on covariance warmup state and contract availability:

    warmup   — t < warmup_weeks finite theta updates; all output fields null
    degraded — warmup complete but strict contracts not satisfied; h_t/rho_t null
    strict   — warmup complete and weekly_h_t contract provided; full tuple

The IIR h_t^lead transform is maintained inside the loop (Option B) so the
state machine stays co-located with the computation and is fully auditable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Mapping

import numpy as np
import pandas as pd

from qqq_cycle.backtest.diagnostics import (
    _semantic_label,
    _state_probabilities,
)
from qqq_cycle.config import ModelConfig, load_config
from qqq_cycle.core.covariance import RobustEWCov2D
from qqq_cycle.core.drift_probe import DriftProbe
from qqq_cycle.core.interpretability import (
    InterpretabilityRecord,
    ModuleHealth,
    build_interpretability,
)
from qqq_cycle.core.micro_layer import (
    MicroIIRState,
    MicroDailyState,
    MicroLayerUnavailableError,
    compute_breadth,
    compute_correlation_concentration,
    compute_micro_score,
    compute_smoothed_weights,
    matured_member_sets,
    should_hold_for_giant_missing_weight,
    update_micro_daily_state,
    update_weekly_micro_iir_state,
    weekly_median_micro,
    z_wrob_156,
)
from qqq_cycle.core.proto_online import (
    PrototypeState,
    initialize_prototypes_from_history,
    update_prototypes,
)
from qqq_cycle.core.risk_layer import RiskScore, blended_state_weight, compute_risk_score
from qqq_cycle.core.state_layer import compute_state_layer
from qqq_cycle.core.stress_layer import compute_stress_layer
from qqq_cycle.data_contracts.constituents import ConstituentStore
from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError, PITAdjustmentEngine
from qqq_cycle.data_contracts.weights import WeightStore

MODE_WARMUP = "warmup"
MODE_DEGRADED = "degraded"
MODE_STRICT = "strict"

# Consecutive weeks below heal_threshold before IIR envelope is reset.
# Derived from model spec §micro: "3-week heal circuit breaker".
_HEAL_CIRCUIT_WEEKS = 3


@dataclass(frozen=True)
class PipelineContracts:
    """Strict input contracts that gate h_t and rho_t computation.

    Phase 6 path: supply weekly_h_t directly (pre-computed or synthetic fixture).
    Phase 7 path: supply pit_engine + constituent_store + weight_store; the
        pipeline calls _compute_weekly_h_t_from_stores() internally.
    If both are supplied, the stores path takes precedence.
    The three boolean flags are auto-derived from the store objects when stores
    are provided; otherwise they must be set explicitly (Phase 6 fixture).
    """

    weekly_h_t: pd.Series | None = None  # index=week_end Timestamps, values=raw h_t floats
    # Phase 7 real stores (optional; trigger _compute_weekly_h_t_from_stores).
    pit_engine: PITAdjustmentEngine | None = None
    constituent_store: ConstituentStore | None = None
    weight_store: WeightStore | None = None
    # Explicit availability flags (Phase 6 fixture compat; overridden when stores present).
    pit_engine_available: bool = False
    constituents_available: bool = False
    weights_available: bool = False


@dataclass(frozen=True)
class PipelineResult:
    """Single-week pipeline output with explicit mode and audit fields.

    Fields are null according to the mode:
        warmup:   all output fields null; strict_contracts_satisfied=None
        degraded: h_t=rho_t=null; degraded_reason non-empty; strict_contracts_satisfied=False
        strict:   full tuple; h_t=raw value; rho_t via IIR h_t_lead
    """

    week_end: str
    k_hat_t: int | None
    p_t: list[float] | None
    s_t: float | None
    h_t: float | None
    rho_t: float | None
    I_t: InterpretabilityRecord | None
    interpretability: dict | None
    mode: str
    degraded_reason: str | None
    strict_contracts_satisfied: bool | None
    backfill_mode: str | None = None
    micro_state_frozen: bool = False
    micro_envelope_internal_state: float | None = None
    micro_breaker_internal_state: str | None = None
    micro_rho_update_state: str | None = None
    contract_source: str | None = None
    strict_gate_passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "week_end": self.week_end,
            "k_hat_t": self.k_hat_t,
            "p_t": json.dumps(self.p_t) if self.p_t is not None else None,
            "s_t": self.s_t,
            "h_t": self.h_t,
            "rho_t": self.rho_t,
            "I_t": (
                asdict(self.I_t)
                if isinstance(self.I_t, InterpretabilityRecord)
                else self.I_t
            ),
            "interpretability": json.dumps(self.interpretability) if self.interpretability else None,
            "mode": self.mode,
            "degraded_reason": self.degraded_reason,
            "strict_contracts_satisfied": self.strict_contracts_satisfied,
            "backfill_mode": self.backfill_mode,
            "micro_state_frozen": self.micro_state_frozen,
            "micro_envelope_internal_state": self.micro_envelope_internal_state,
            "micro_breaker_internal_state": self.micro_breaker_internal_state,
            "micro_rho_update_state": self.micro_rho_update_state,
            "contract_source": self.contract_source,
            "strict_gate_passed": self.strict_gate_passed,
        }


def _check_strict_gate(
    contracts: PipelineContracts | None,
) -> tuple[bool, str | None]:
    """Return (can_compute_h_t, degraded_reason).

    can_compute_h_t is True only if contracts is not None and weekly_h_t is provided.
    degraded_reason is None when can_compute_h_t is True.
    """
    if contracts is None:
        return False, "no contracts provided: h_t/rho_t unavailable"
    if contracts.weekly_h_t is None:
        missing = []
        if not contracts.pit_engine_available:
            missing.append("pit_engine")
        if not contracts.constituents_available:
            missing.append("historical_constituents")
        if not contracts.weights_available:
            missing.append("historical_weights")
        suffix = ", ".join(missing) if missing else "weekly_h_t not provided"
        return False, f"strict contracts not satisfied: {suffix}"
    return True, None


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _value_or_nan(value: float | None) -> float:
    return float(value) if value is not None else float("nan")


def _empty_weekly_h_t(weekly_index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(float("nan"), index=weekly_index, name="h_t")


def _strict_pit_engine_available(engine: object | None) -> bool:
    """Return whether `engine` satisfies production strict PIT micro semantics."""

    return (
        isinstance(engine, PITAdjustmentEngine)
        and getattr(engine, "asof_semantics", None) == "strict_pit"
    )


def _delta4(frame: pd.DataFrame, week_end: pd.Timestamp, column: str) -> float:
    if week_end not in frame.index:
        return float("nan")
    loc = frame.index.get_loc(week_end)
    if isinstance(loc, slice) or isinstance(loc, np.ndarray):
        return float("nan")
    if int(loc) < 4:
        return float("nan")
    current = _safe_float(frame.iloc[int(loc)][column])
    previous = _safe_float(frame.iloc[int(loc) - 4][column])
    if current is None or previous is None:
        return float("nan")
    return current - previous


def _build_audit_interpretability(
    week_end: pd.Timestamp,
    state: pd.DataFrame,
    stress_frame: pd.DataFrame,
    drift_frame: pd.DataFrame,
    *,
    omega_t: float | None,
    s_t: float | None,
    n_t: float | None,
    h_t: float | None,
    h_t_available: bool,
    rho_t_available: bool,
    config: ModelConfig,
    state_ok: bool = True,
) -> InterpretabilityRecord:
    """Build the model-spec I_t audit object from point-in-time layer outputs."""

    L_t = _safe_float(state.at[week_end, "L"]) if week_end in state.index else None
    T_t = _safe_float(state.at[week_end, "T"]) if week_end in state.index else None
    P_t = _safe_float(state.at[week_end, "P"]) if week_end in state.index else None
    E_t = _safe_float(state.at[week_end, "E"]) if week_end in state.index else None
    g_raw = _safe_float(stress_frame.at[week_end, "g_raw"]) if week_end in stress_frame.index else None
    g_stress = _safe_float(stress_frame.at[week_end, "g_stress"]) if week_end in stress_frame.index else None
    delta_abs = (
        _safe_float(drift_frame.at[week_end, "drift_probe_raw"])
        if week_end in drift_frame.index
        else None
    )
    module_health = ModuleHealth(
        h_macro=int(L_t is not None and T_t is not None and P_t is not None),
        h_exo=int(E_t is not None),
        h_micro=int(h_t_available),
        h_state=int(bool(state_ok)),
    )
    return build_interpretability(
        L_t=_value_or_nan(L_t),
        T_t=_value_or_nan(T_t),
        P_t=_value_or_nan(P_t),
        delta4_L_t=_delta4(state, week_end, "L"),
        delta4_T_t=_delta4(state, week_end, "T"),
        delta4_P_t=_delta4(state, week_end, "P"),
        g_tilde=_value_or_nan(g_raw),
        e_tilde=_value_or_nan(E_t),
        b_tilde=float("nan"),
        c_tilde=float("nan"),
        omega_t=_value_or_nan(omega_t),
        s_t=_value_or_nan(s_t),
        n_t=_value_or_nan(n_t),
        eta_t=float("nan"),
        is_rule_week=False,
        has_constituent_change=False,
        data_contaminated=not rho_t_available,
        v60_count=0,
        universe_count=0,
        delta_abs_raw=_value_or_nan(delta_abs),
        d_state=_value_or_nan(delta_abs),
        g_stress=_value_or_nan(g_stress),
        micro_raw=_value_or_nan(h_t),
        module_health=module_health,
        lambda_rho=config.risk.lambda_rho,
        theta_drift_hi=config.drift.theta_hi,
    )


def _build_interpretability(
    week_end: pd.Timestamp,
    state: pd.DataFrame,
    stress_frame: pd.DataFrame,
    drift_frame: pd.DataFrame,
    k_hat_t: int | None,
    p_t: list[float] | None,
    h_t: float | None,
    rho_t: float | None,
) -> dict:
    """Build a minimal interpretability dict for one week."""
    row: dict[str, Any] = {}
    for col in ("L", "T", "P", "E", "H", "I"):
        row[col] = _safe_float(state.at[week_end, col]) if week_end in state.index else None
    for col in ("d", "a", "g_raw", "g_stress", "s"):
        row[col] = _safe_float(stress_frame.at[week_end, col]) if week_end in stress_frame.index else None
    row["drift_probe_raw"] = (
        _safe_float(drift_frame.at[week_end, "drift_probe_raw"])
        if week_end in drift_frame.index
        else None
    )
    row["drift_flag"] = (
        int(drift_frame.at[week_end, "drift_flag"])
        if week_end in drift_frame.index and pd.notna(drift_frame.at[week_end, "drift_flag"])
        else 0
    )
    row["k_hat_t"] = k_hat_t
    row["p_t"] = p_t
    row["h_t"] = h_t
    row["rho_t"] = rho_t
    return row


def _compute_daily_micro_frame(
    trading_days: pd.DatetimeIndex,
    contracts: PipelineContracts,
) -> pd.DataFrame:
    """Compute daily b_tau/c_tau from strict PIT stores.

    Inputs:
        trading_days: Daily decision dates to scan in chronological order.
        contracts: Store-backed contracts with strict PIT price adjustment.

    Output:
        DataFrame indexed by trade date with daily `b_tau`, `c_tau`, and
        `z_weight` for weekly robust standardization.

    Time semantics:
        Each row uses the current PIT constituent snapshot, previous trading
        day's visible weights through the five-day smoothed lag, and PIT price
        windows whose `asof` is the current daily EOD timestamp.

    Failure modes:
        MicroLayerUnavailableError: strict PIT adjustment is absent or a mature
        member's mandatory PIT price window cannot be reconstructed.
    """
    pit_engine = contracts.pit_engine
    constituent_store = contracts.constituent_store
    weight_store = contracts.weight_store
    if not _strict_pit_engine_available(pit_engine):
        raise MicroLayerUnavailableError("strict PITAdjustmentEngine is required")
    if constituent_store is None or weight_store is None:
        raise MicroLayerUnavailableError("constituent and weight stores are required")

    micro_state = MicroDailyState.empty()
    daily_records: list[dict] = []
    previous_day_weights: dict[str, float] | None = None
    previous_smoothed: dict[str, float] = {}
    previous_members: frozenset[str] | None = None
    last_b_tau = float("nan")
    last_c_tau = float("nan")

    for day in trading_days:
        trade_ts = pd.Timestamp(day).normalize()
        # Use end-of-day asof so same-day EOD data (timestamped T16:00) is visible.
        asof_eod = trade_ts + pd.Timedelta(hours=23, minutes=59, seconds=59)
        try:
            snapshot = constituent_store.get_snapshot(trade_ts, asof=asof_eod)
        except DataNotAvailableError:
            previous_day_weights = None
            previous_members = None
            continue
        try:
            raw_weights = weight_store.get_weights(trade_ts, asof=asof_eod)
        except DataNotAvailableError:
            previous_day_weights = None
            previous_members = None
            continue

        has_constituent_change = (
            previous_members is not None and snapshot.members != previous_members
        )
        micro_state = update_micro_daily_state(
            micro_state, snapshot.members, trade_ts
        )
        if previous_day_weights is None:
            previous_members = snapshot.members
            previous_day_weights = dict(raw_weights)
            daily_records.append(
                {"date": trade_ts, "b_tau": np.nan, "c_tau": np.nan, "z_weight": 1.0}
            )
            continue

        smoothed_weights = compute_smoothed_weights(
            previous_smoothed,
            previous_day_weights,
            is_rule_window=has_constituent_change,
        )
        previous_smoothed = smoothed_weights
        micro_state = micro_state.with_smoothed_weights(smoothed_weights)
        missing_decision = should_hold_for_giant_missing_weight(micro_state)
        v20, v60 = matured_member_sets(micro_state)
        z_weight = (
            0.3
            if (
                micro_state.data_contaminated
                or has_constituent_change
                or missing_decision.data_contaminated
            )
            else 1.0
        )

        if missing_decision.hold_micro_recompute:
            b_tau = last_b_tau
            c_tau = last_c_tau
        else:
            b_tau = compute_breadth(
                members=v20,
                smoothed_weights=smoothed_weights,
                trade_date=trade_ts,
                pit_engine=pit_engine,
                asof=asof_eod,
            )
            price_windows: dict[str, pd.Series] = {}
            for ticker in sorted(v60):
                w = float(smoothed_weights.get(ticker, 0.0))
                if w <= 0.0:
                    continue
                try:
                    window = pit_engine.get_adjusted_window(
                        ticker, trade_ts, 60, asof=asof_eod
                    )
                except DataNotAvailableError as exc:
                    raise MicroLayerUnavailableError(
                        f"missing PIT 60-day adjusted window for {ticker} on {trade_ts.date()}"
                    ) from exc
                if len(pd.Series(window).dropna()) < 60:
                    raise MicroLayerUnavailableError(
                        f"need 60 PIT adjusted closes for {ticker}; got {len(pd.Series(window).dropna())}"
                    )
                price_windows[ticker] = window
            c_tau = compute_correlation_concentration(
                members=v60,
                smoothed_weights=smoothed_weights,
                price_windows=price_windows,
            )
            if np.isfinite(b_tau):
                last_b_tau = b_tau
            if np.isfinite(c_tau):
                last_c_tau = c_tau

        daily_records.append(
            {"date": trade_ts, "b_tau": b_tau, "c_tau": c_tau, "z_weight": z_weight}
        )
        previous_members = snapshot.members
        previous_day_weights = dict(raw_weights)

    if not daily_records:
        return pd.DataFrame(columns=["b_tau", "c_tau", "z_weight"])
    return pd.DataFrame(daily_records).set_index("date")


def _compute_weekly_h_t_from_stores(
    weekly_index: pd.DatetimeIndex,
    contracts: PipelineContracts,
    config: ModelConfig,
) -> pd.Series:
    """Pre-compute weekly h_t from real daily micro stores.

    Runs the daily micro loop over all business days in the weekly_index range,
    resamples to weekly Friday medians, applies z_wrob_156 standardization, and
    applies compute_micro_score to produce h_t per week.  Returns a pd.Series
    aligned to weekly_index; weeks without sufficient data carry NaN.
    """
    del config

    start = weekly_index.min()
    end = weekly_index.max()
    trading_days = pd.bdate_range(start=start, end=end)

    try:
        daily_df = _compute_daily_micro_frame(trading_days, contracts)
    except MicroLayerUnavailableError:
        return _empty_weekly_h_t(weekly_index)

    if daily_df.empty:
        return _empty_weekly_h_t(weekly_index)
    weekly_bc = weekly_median_micro(daily_df)

    weekly_z_weights = daily_df["z_weight"].resample("W-FRI", label="right", closed="right").min()
    weekly_z_weights = weekly_z_weights.reindex(weekly_bc.index).fillna(1.0)
    b_z = z_wrob_156(weekly_bc["b_wk"], weights=weekly_z_weights)
    c_z = z_wrob_156(weekly_bc["c_wk"], weights=weekly_z_weights)

    h_t_weekly = pd.Series(float("nan"), index=weekly_bc.index, name="h_t")
    for idx in weekly_bc.index:
        bz = b_z.get(idx, float("nan"))
        cz = c_z.get(idx, float("nan"))
        if np.isfinite(bz) and np.isfinite(cz):
            score = compute_micro_score(bz, cz)
            h_t_weekly[idx] = score.h_t

    return h_t_weekly.reindex(weekly_index)


def run_pipeline(
    weekly_macro_inputs: pd.DataFrame,
    contracts: PipelineContracts | None = None,
    config: ModelConfig | None = None,
    backfill_modes: Mapping[str | pd.Timestamp, str] | None = None,
    contract_sources: Mapping[str | pd.Timestamp, str] | None = None,
    strict_gate_passed: Mapping[str | pd.Timestamp, bool] | None = None,
) -> list[PipelineResult]:
    """Run the end-to-end pipeline and return one PipelineResult per week.

    Modes per row:
        warmup   — cov.is_warm() is False; all outputs null; IIR state still advances
        degraded — post-warmup; contracts.weekly_h_t is None; h_t/rho_t null
        strict   — post-warmup; weekly_h_t provided; full tuple via IIR h_t_lead

    Args:
        weekly_macro_inputs: DataFrame indexed by week-end dates with columns
            DFII10, DGS2, BAMLH0A0HYM2, NFCI, VIXCLS, USEPUINDXD, AI_GPR, QQQ.
        contracts: Optional strict input contracts. None → degraded mode.
        config: Optional ModelConfig. If None, loads from model_v22.yaml.
        backfill_modes: Optional week_end to controlled backfill mode mapping.
            A week with mode degraded_backfill freezes micro IIR state.

    Returns:
        List of PipelineResult, one per row in weekly_macro_inputs.

    Raises:
        ValueError: If strict contracts are claimed but h_t lookup fails for a
            post-warmup week that has a non-NaN index entry. This is a hard
            error — no silent fallback.
    """
    if config is None:
        config = load_config()

    # Phase 7: if real stores are provided, pre-compute weekly_h_t from daily micro loop.
    # The stores path takes precedence over a pre-supplied weekly_h_t series.
    if (
        contracts is not None
        and contracts.pit_engine is not None
        and contracts.constituent_store is not None
        and contracts.weight_store is not None
    ):
        strict_pit_available = _strict_pit_engine_available(contracts.pit_engine)
        computed_h_t = (
            _compute_weekly_h_t_from_stores(weekly_macro_inputs.index, contracts, config)
            if strict_pit_available
            else None
        )
        contracts = PipelineContracts(
            weekly_h_t=computed_h_t,
            pit_engine=contracts.pit_engine,
            constituent_store=contracts.constituent_store,
            weight_store=contracts.weight_store,
            pit_engine_available=strict_pit_available,
            constituents_available=True,
            weights_available=True,
        )

    can_compute_h_t, degraded_reason = _check_strict_gate(contracts)
    normalized_backfill_modes = _normalize_backfill_modes(backfill_modes)
    normalized_contract_sources = _normalize_text_mapping(contract_sources)
    normalized_strict_gate = _normalize_bool_mapping(strict_gate_passed)

    # Precompute all layer outputs (batch, not per-row) before the routing loop.
    state = compute_state_layer(weekly_macro_inputs)
    theta = state[["H", "I"]]
    stress_result = compute_stress_layer(theta, state["E"])
    stress_frame = stress_result.frame
    drift_frame = DriftProbe(
        theta_lo=config.drift.theta_lo,
        theta_hi=config.drift.theta_hi,
    ).compute(weekly_macro_inputs)

    finite_theta = theta.dropna()
    if len(finite_theta) < 20:
        # Cannot even initialize covariance; return all-warmup rows.
        return [
            PipelineResult(
                week_end=ts.strftime("%Y-%m-%d"),
                k_hat_t=None,
                p_t=None,
                s_t=None,
                h_t=None,
                rho_t=None,
                I_t=None,
                interpretability=None,
                mode=MODE_WARMUP,
                degraded_reason=None,
                strict_contracts_satisfied=None,
                backfill_mode=normalized_backfill_modes.get(pd.Timestamp(ts).strftime("%Y-%m-%d")),
                contract_source=normalized_contract_sources.get(pd.Timestamp(ts).strftime("%Y-%m-%d")),
                strict_gate_passed=normalized_strict_gate.get(pd.Timestamp(ts).strftime("%Y-%m-%d"), False),
            )
            for ts in theta.index
        ]

    cov = RobustEWCov2D(warmup_weeks=config.warmup_weeks)
    cov_state = cov.initialize_from_history(finite_theta.iloc[:20].to_numpy())
    proto: PrototypeState | None = None
    proto_seed: list[np.ndarray] = []

    # IIR h_t^lead state — initialized BEFORE the loop (Option B).
    # These must not be reset inside the loop; they carry state across weeks.
    h_t_lead_prev: float = 0.5
    heal_count: int = 0

    omega_state = np.asarray(config.risk.omega_state, dtype=float)

    results: list[PipelineResult] = []

    for week_end, theta_row in theta.iterrows():
        x = theta_row.to_numpy(dtype=float)

        if not cov.is_warm(cov_state):
            # Warmup: advance internal state but suppress all outputs.
            # Do NOT read contracts.weekly_h_t — warmup rows must not consume it.
            if np.all(np.isfinite(x)):
                cov_state = cov.update(cov_state, x)
                proto_seed.append(x)
            else:
                cov_state = cov.update(cov_state, np.array([np.nan, np.nan]))
            results.append(
                PipelineResult(
                    week_end=week_end.strftime("%Y-%m-%d"),
                    k_hat_t=None,
                    p_t=None,
                    s_t=None,  # s_t MUST be null during warmup — not read from stress_frame
                    h_t=None,
                    rho_t=None,
                    I_t=None,
                    interpretability=None,
                    mode=MODE_WARMUP,
                    degraded_reason=None,
                    strict_contracts_satisfied=None,
                )
            )
            continue

        # Post-warmup: compute state probabilities.
        k_hat_t: int | None = None
        p_t: list[float] | None = None

        if np.all(np.isfinite(x)):
            if proto is None and len(proto_seed) >= config.warmup_weeks:
                proto = initialize_prototypes_from_history(np.asarray(proto_seed))
            if proto is not None:
                prev_cov = cov_state.cov_reg.copy()
                cov_state = cov.update(cov_state, x)
                proto_result = update_prototypes(
                    proto, x, cov_state.mean, prev_cov, cov_state.cov_reg, len(results)
                )
                proto = proto_result.state
                probs = _state_probabilities(x, proto, cov_state.cov_reg)
                k_hat_t = int(np.argmax(probs))
                p_t = [float(p) for p in probs]
            else:
                cov_state = cov.update(cov_state, x)
        else:
            cov_state = cov.update(cov_state, np.array([np.nan, np.nan]))

        # s_t: read from pre-computed stress frame (safe post-warmup).
        s_t = (
            _safe_float(stress_frame.at[week_end, "s"])
            if week_end in stress_frame.index
            else None
        )

        # Strict gate: h_t and rho_t.
        h_t: float | None = None
        rho_t: float | None = None
        omega_t: float | None = None
        n_t: float | None = None
        strict_contracts_satisfied: bool | None = False
        backfill_mode = normalized_backfill_modes.get(pd.Timestamp(week_end).strftime("%Y-%m-%d"))
        contract_source = normalized_contract_sources.get(pd.Timestamp(week_end).strftime("%Y-%m-%d"))
        strict_gate_for_week = normalized_strict_gate.get(pd.Timestamp(week_end).strftime("%Y-%m-%d"), False)
        micro_iir_state = MicroIIRState(
            h_t_lead_prev=h_t_lead_prev,
            heal_count=heal_count,
            envelope_internal_state=h_t_lead_prev,
            breaker_internal_state="active" if heal_count else "inactive",
            rho_update_state="prior_pipeline_state",
        )

        if backfill_mode in {"degraded_backfill", "block"}:
            micro_iir_state = update_weekly_micro_iir_state(
                micro_iir_state,
                h_t_raw=None,
                backfill_mode=backfill_mode,
                delta=config.micro.iir_delta,
                theta_heal=config.micro.heal_threshold,
                heal_weeks=_HEAL_CIRCUIT_WEEKS,
            )
            h_t_lead_prev = micro_iir_state.h_t_lead_prev
            heal_count = micro_iir_state.heal_count
            strict_contracts_satisfied = False
        elif can_compute_h_t:
            h_t_series = contracts.weekly_h_t  # type: ignore[union-attr]
            if week_end in h_t_series.index:
                raw_val = h_t_series.loc[week_end]
                h_t_raw = _safe_float(raw_val)
            else:
                h_t_raw = None

            # Advance IIR state every week post-warmup (decay on missing observation)
            micro_iir_state = update_weekly_micro_iir_state(
                micro_iir_state,
                h_t_raw=h_t_raw,
                backfill_mode=backfill_mode,
                delta=config.micro.iir_delta,
                theta_heal=config.micro.heal_threshold,
                heal_weeks=_HEAL_CIRCUIT_WEEKS,
            )
            h_t_lead = micro_iir_state.h_t_lead_prev
            h_t_lead_prev = micro_iir_state.h_t_lead_prev
            heal_count = micro_iir_state.heal_count

            if h_t_raw is not None:
                h_t = h_t_raw

                # Compute omega_t for rho_t.
                drift_raw = (
                    _safe_float(drift_frame.at[week_end, "drift_probe_raw"])
                    if week_end in drift_frame.index
                    else None
                )
                delta_abs = abs(drift_raw) if drift_raw is not None else 0.0

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

            strict_contracts_satisfied = bool(
                h_t is not None
                and contracts.pit_engine_available  # type: ignore[union-attr]
                and contracts.constituents_available  # type: ignore[union-attr]
                and contracts.weights_available  # type: ignore[union-attr]
            )

        mode = MODE_STRICT if h_t is not None else MODE_DEGRADED
        if mode == MODE_STRICT:
            row_degraded_reason = None
        elif backfill_mode == "degraded_backfill":
            row_degraded_reason = "controlled degraded_backfill: micro IIR state frozen"
        elif backfill_mode == "block":
            row_degraded_reason = "controlled block: micro IIR state held"
        elif can_compute_h_t and h_t is None:
            # Weekly h_t series was provided but NaN for this week (e.g., micro
            # data not yet available or z_wrob window not yet satisfied).
            row_degraded_reason = "h_t unavailable for this week: micro data window not satisfied"
        else:
            row_degraded_reason = degraded_reason

        interp = _build_interpretability(
            week_end, state, stress_frame, drift_frame, k_hat_t, p_t, h_t, rho_t
        )
        I_t = _build_audit_interpretability(
            week_end,
            state,
            stress_frame,
            drift_frame,
            omega_t=omega_t,
            s_t=s_t,
            n_t=n_t,
            h_t=h_t,
            h_t_available=h_t is not None,
            rho_t_available=rho_t is not None,
            config=config,
            state_ok=bool(
                cov_state.state_ok
                and stress_result.velocity_cov_state.state_ok
                and stress_result.acceleration_cov_state.state_ok
            ),
        )

        results.append(
            PipelineResult(
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
                strict_contracts_satisfied=strict_contracts_satisfied,
                backfill_mode=backfill_mode,
                micro_state_frozen=micro_iir_state.micro_state_frozen,
                micro_envelope_internal_state=micro_iir_state.envelope_internal_state,
                micro_breaker_internal_state=micro_iir_state.breaker_internal_state,
                micro_rho_update_state=micro_iir_state.rho_update_state,
                contract_source=contract_source,
                strict_gate_passed=strict_gate_for_week,
            )
        )

    return results


def _normalize_backfill_modes(
    backfill_modes: Mapping[str | pd.Timestamp, str] | None,
) -> dict[str, str]:
    if not backfill_modes:
        return {}
    return {
        pd.Timestamp(week_end).strftime("%Y-%m-%d"): str(mode)
        for week_end, mode in backfill_modes.items()
    }


def _normalize_text_mapping(
    values: Mapping[str | pd.Timestamp, str] | None,
) -> dict[str, str]:
    if not values:
        return {}
    return {pd.Timestamp(key).strftime("%Y-%m-%d"): str(value) for key, value in values.items()}


def _normalize_bool_mapping(
    values: Mapping[str | pd.Timestamp, bool] | None,
) -> dict[str, bool]:
    if not values:
        return {}
    return {pd.Timestamp(key).strftime("%Y-%m-%d"): bool(value) for key, value in values.items()}


def results_to_frame(results: list[PipelineResult]) -> pd.DataFrame:
    """Convert a list of PipelineResult to a DataFrame with one row per week."""
    return pd.DataFrame([r.to_dict() for r in results])
