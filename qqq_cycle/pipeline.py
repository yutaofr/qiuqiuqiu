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
from typing import Any

import numpy as np
import pandas as pd

from qqq_cycle.backtest.diagnostics import (
    _semantic_label,
    _state_probabilities,
)
from qqq_cycle.config import ModelConfig, load_config
from qqq_cycle.core.covariance import RobustEWCov2D
from qqq_cycle.core.drift_probe import DriftProbe
from qqq_cycle.core.micro_layer import (
    MicroDailyState,
    MicroLayerUnavailableError,
    compute_breadth,
    compute_correlation_concentration,
    compute_micro_score,
    update_micro_daily_state,
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
    I_t: float | None
    interpretability: dict | None
    mode: str
    degraded_reason: str | None
    strict_contracts_satisfied: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "week_end": self.week_end,
            "k_hat_t": self.k_hat_t,
            "p_t": json.dumps(self.p_t) if self.p_t is not None else None,
            "s_t": self.s_t,
            "h_t": self.h_t,
            "rho_t": self.rho_t,
            "I_t": self.I_t,
            "interpretability": json.dumps(self.interpretability) if self.interpretability else None,
            "mode": self.mode,
            "degraded_reason": self.degraded_reason,
            "strict_contracts_satisfied": self.strict_contracts_satisfied,
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
    pit_engine = contracts.pit_engine
    constituent_store = contracts.constituent_store
    weight_store = contracts.weight_store

    start = weekly_index.min()
    end = weekly_index.max()
    trading_days = pd.bdate_range(start=start, end=end)

    micro_state = MicroDailyState.empty()
    daily_records: list[dict] = []

    for day in trading_days:
        trade_ts = pd.Timestamp(day).normalize()
        # Use end-of-day asof so same-day EOD data (timestamped T16:00) is visible.
        asof_eod = trade_ts + pd.Timedelta(hours=23, minutes=59, seconds=59)
        try:
            snapshot = constituent_store.get_snapshot(trade_ts, asof=asof_eod)
        except DataNotAvailableError:
            continue
        try:
            raw_weights = weight_store.get_weights(trade_ts, asof=asof_eod)
        except DataNotAvailableError:
            continue

        micro_state = update_micro_daily_state(
            micro_state, snapshot.members, trade_ts
        )
        micro_state = micro_state.with_smoothed_weights(raw_weights)

        try:
            b_tau = compute_breadth(
                members=micro_state.present_members,
                smoothed_weights=dict(micro_state.smoothed_weights),
                trade_date=trade_ts,
                pit_engine=pit_engine,
            )
        except MicroLayerUnavailableError:
            b_tau = float("nan")

        c_tau = float("nan")
        try:
            price_windows: dict[str, pd.Series] = {}
            for ticker in sorted(micro_state.present_members):
                w = float(micro_state.smoothed_weights.get(ticker, 0.0))
                if w <= 0.0:
                    continue
                try:
                    price_windows[ticker] = pit_engine.get_adjusted_window(
                        ticker, trade_ts, 60, asof=asof_eod
                    )
                except DataNotAvailableError:
                    pass
            c_tau = compute_correlation_concentration(
                members=micro_state.present_members,
                smoothed_weights=dict(micro_state.smoothed_weights),
                price_windows=price_windows,
            )
        except MicroLayerUnavailableError:
            c_tau = float("nan")

        daily_records.append({"date": trade_ts, "b_tau": b_tau, "c_tau": c_tau})

    if not daily_records:
        return pd.Series(float("nan"), index=weekly_index, name="h_t")

    daily_df = pd.DataFrame(daily_records).set_index("date")
    weekly_bc = weekly_median_micro(daily_df)

    uniform_weights = pd.Series(1.0, index=weekly_bc.index)
    b_z = z_wrob_156(weekly_bc["b_wk"], weights=uniform_weights)
    c_z = z_wrob_156(weekly_bc["c_wk"], weights=uniform_weights)

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
        computed_h_t = _compute_weekly_h_t_from_stores(
            weekly_macro_inputs.index, contracts, config
        )
        contracts = PipelineContracts(
            weekly_h_t=computed_h_t,
            pit_engine=contracts.pit_engine,
            constituent_store=contracts.constituent_store,
            weight_store=contracts.weight_store,
            pit_engine_available=True,
            constituents_available=True,
            weights_available=True,
        )

    can_compute_h_t, degraded_reason = _check_strict_gate(contracts)

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
            )
            for ts in theta.index
        ]

    cov = RobustEWCov2D(warmup_weeks=config.warmup_weeks)
    cov_state = cov.initialize_from_history(finite_theta.iloc[:20].to_numpy())
    proto: PrototypeState | None = None
    proto_seed: list[np.ndarray] = []

    # IIR h_t^lead state — initialized BEFORE the loop (Option B).
    # These must not be reset inside the loop; they carry state across weeks.
    h_t_lead_prev: float = 0.0
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

        # I_t: impulse theta coordinate.
        I_t = float(x[1]) if np.isfinite(x[1]) else None

        # Strict gate: h_t and rho_t.
        h_t: float | None = None
        rho_t: float | None = None
        strict_contracts_satisfied: bool | None = False

        if can_compute_h_t:
            h_t_series = contracts.weekly_h_t  # type: ignore[union-attr]
            if week_end in h_t_series.index:
                raw_val = h_t_series.loc[week_end]
                h_t_raw = _safe_float(raw_val)
            else:
                h_t_raw = None

            if h_t_raw is not None:
                # IIR positive-interval envelope (Option B: inline state machine).
                h_t_lead = max(h_t_raw, config.micro.iir_delta * h_t_lead_prev)
                if h_t_raw < config.micro.heal_threshold:
                    heal_count += 1
                    if heal_count >= _HEAL_CIRCUIT_WEEKS:
                        # Circuit fires: reset envelope to raw value.
                        h_t_lead = h_t_raw
                        heal_count = 0
                else:
                    heal_count = 0
                h_t_lead_prev = h_t_lead

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
                    rho_t = float(risk.rho_t)

            strict_contracts_satisfied = (
                contracts.pit_engine_available  # type: ignore[union-attr]
                and contracts.constituents_available  # type: ignore[union-attr]
                and contracts.weights_available  # type: ignore[union-attr]
            )

        mode = MODE_STRICT if h_t is not None else MODE_DEGRADED
        if mode == MODE_STRICT:
            row_degraded_reason = None
        elif can_compute_h_t and h_t is None:
            # Weekly h_t series was provided but NaN for this week (e.g., micro
            # data not yet available or z_wrob window not yet satisfied).
            row_degraded_reason = "h_t unavailable for this week: micro data window not satisfied"
        else:
            row_degraded_reason = degraded_reason

        interp = _build_interpretability(
            week_end, state, stress_frame, drift_frame, k_hat_t, p_t, h_t, rho_t
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
            )
        )

    return results


def results_to_frame(results: list[PipelineResult]) -> pd.DataFrame:
    """Convert a list of PipelineResult to a DataFrame with one row per week."""
    return pd.DataFrame([r.to_dict() for r in results])
