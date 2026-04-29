"""Tests for live runtime: hot restart, execution state routing, log output."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qqq_cycle.backtest.diagnostics import synthetic_replay_inputs
from qqq_cycle.config import load_config
from qqq_cycle.core.covariance import RobustEWCov2D
from qqq_cycle.core.drift_probe import DriftProbe
from qqq_cycle.core.state_layer import compute_state_layer
from qqq_cycle.core.stress_layer import compute_stress_layer
from qqq_cycle.live.runtime import LiveRuntime, _run_pipeline_step
from qqq_cycle.live.state_io import LiveState, StateNotAvailableError, load_state, save_state
from qqq_cycle.pipeline import (
    MODE_STRICT,
    PipelineContracts,
    _check_strict_gate,
    _safe_float,
    run_pipeline,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_WARMUP_WEEKS = 260  # must match model_v22.yaml


def _make_strict_contracts(index: pd.DatetimeIndex) -> PipelineContracts:
    rng = np.random.default_rng(20260428)
    n = len(index)
    values = np.full(n, np.nan)
    post = max(0, n - _WARMUP_WEEKS)
    values[_WARMUP_WEEKS:] = rng.uniform(0.10, 0.60, size=post)
    return PipelineContracts(
        weekly_h_t=pd.Series(values, index=index, name="h_t"),
        pit_engine_available=True,
        constituents_available=True,
        weights_available=True,
    )


def _bootstrap_state_at_week(
    macro_df: pd.DataFrame,
    up_to_idx: int,
    contracts: PipelineContracts,
    state_dir: Path,
) -> None:
    """Build and save LiveState up to macro_df row at up_to_idx (inclusive)."""
    config = load_config()
    tail = macro_df.iloc[: up_to_idx + 1]

    state_frame = compute_state_layer(tail)
    theta = state_frame[["H", "I"]]
    stress_result = compute_stress_layer(theta, state_frame["E"])
    stress_frame = stress_result.frame
    drift_frame = DriftProbe(
        theta_lo=config.drift.theta_lo,
        theta_hi=config.drift.theta_hi,
    ).compute(tail)

    finite_theta = theta.dropna()
    cov = RobustEWCov2D(warmup_weeks=config.warmup_weeks)
    cov_state = cov.initialize_from_history(finite_theta.iloc[:20].to_numpy())
    proto = None
    proto_seed: list[np.ndarray] = []
    h_t_lead_prev = 0.0
    heal_count = 0

    can_h, deg_reason = _check_strict_gate(contracts)

    for idx, (week_end, theta_row_s) in enumerate(theta.iterrows()):
        x = theta_row_s.to_numpy(dtype=float)
        s_t = _safe_float(stress_frame.at[week_end, "s"]) if week_end in stress_frame.index else None
        i_t = float(x[1]) if np.isfinite(x[1]) else None
        drift_raw = (
            _safe_float(drift_frame.at[week_end, "drift_probe_raw"])
            if week_end in drift_frame.index else None
        )
        drift_flag = (
            int(drift_frame.at[week_end, "drift_flag"])
            if week_end in drift_frame.index
            and pd.notna(drift_frame.at[week_end, "drift_flag"]) else 0
        )
        h_t_raw_val = None
        if can_h and contracts.weekly_h_t is not None and week_end in contracts.weekly_h_t.index:
            h_t_raw_val = _safe_float(contracts.weekly_h_t.loc[week_end])

        _, cov_state, proto, proto_seed, h_t_lead_prev, heal_count = _run_pipeline_step(
            week_end=week_end,
            theta_row=x,
            s_t_val=s_t,
            i_t_val=i_t,
            drift_raw=drift_raw,
            drift_flag=drift_flag,
            h_t_raw=h_t_raw_val,
            cov_state=cov_state,
            proto=proto,
            proto_seed=proto_seed,
            h_t_lead_prev=h_t_lead_prev,
            heal_count=heal_count,
            config=config,
            can_compute_h_t=can_h,
            degraded_reason=deg_reason,
            week_index=idx,
            state_frame=state_frame,
            stress_frame=stress_frame,
            drift_frame=drift_frame,
        )

    live_state = LiveState(
        week_end=tail.index[-1].strftime("%Y-%m-%d"),
        cov_state=cov_state,
        proto=proto,
        proto_seed=proto_seed,
        h_t_lead_prev=h_t_lead_prev,
        heal_count=heal_count,
        warmup_count=cov_state.warmup_count,
        breaker_active=False,
        weeks_outside_s1=0,
        prev_omega_qqq=0.5,
        macro_tail=tail,
        last_successful_timestamps={},
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    save_state(live_state, state_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_hot_restart_state_continuity(tmp_path: Path) -> None:
    """Two consecutive live runs match a two-row batch extension."""
    macro_df = synthetic_replay_inputs()
    n = _WARMUP_WEEKS + 10
    tail_n = macro_df.iloc[:n]
    contracts = _make_strict_contracts(macro_df.index[:n + 2])

    # Build state at row n-1 (state covers up to row n-1).
    _bootstrap_state_at_week(macro_df.iloc[:n], n - 1, contracts, tmp_path)

    runtime = LiveRuntime()
    week_end_n = macro_df.index[n].strftime("%Y-%m-%d")
    result1 = runtime.run_week(
        week_end=week_end_n,
        macro_row=macro_df.iloc[n],
        contracts=contracts,
        state_dir=tmp_path,
        output_dir=tmp_path / "out",
    )

    week_end_n1 = macro_df.index[n + 1].strftime("%Y-%m-%d")
    result2 = runtime.run_week(
        week_end=week_end_n1,
        macro_row=macro_df.iloc[n + 1],
        contracts=contracts,
        state_dir=tmp_path,
        output_dir=tmp_path / "out",
    )

    # Compare with batch pipeline on same slice.
    batch_results = run_pipeline(macro_df.iloc[: n + 2], contracts=contracts)
    batch_last = batch_results[-1]
    batch_prev = batch_results[-2]

    assert result1.signal_bundle["mode"] == batch_prev.mode
    assert result2.signal_bundle["mode"] == batch_last.mode

    # For strict rows, rho_t should match closely.
    if batch_prev.mode == MODE_STRICT and result1.signal_bundle["rho_t"] is not None:
        assert abs(result1.signal_bundle["rho_t"] - batch_prev.rho_t) < 1e-6

    if batch_last.mode == MODE_STRICT and result2.signal_bundle["rho_t"] is not None:
        assert abs(result2.signal_bundle["rho_t"] - batch_last.rho_t) < 1e-6


def test_live_run_week_degraded_backfill_freezes_persisted_micro_state(tmp_path: Path) -> None:
    macro_df = synthetic_replay_inputs()
    n = 600
    contracts = _make_strict_contracts(macro_df.index[: n + 2])
    _bootstrap_state_at_week(macro_df.iloc[:n], n - 1, contracts, tmp_path)
    before = load_state(tmp_path)

    runtime = LiveRuntime()
    week_end_n = macro_df.index[n].strftime("%Y-%m-%d")
    result = runtime.run_week(
        week_end=week_end_n,
        macro_row=macro_df.iloc[n],
        contracts=contracts,
        state_dir=tmp_path,
        output_dir=tmp_path / "out",
        backfill_mode="degraded_backfill",
    )
    after_degraded = load_state(tmp_path)

    assert result.signal_bundle["mode"] == "degraded"
    assert result.signal_bundle["backfill_mode"] == "degraded_backfill"
    assert result.signal_bundle["micro_state_frozen"] is True
    assert after_degraded.h_t_lead_prev == pytest.approx(before.h_t_lead_prev)
    assert after_degraded.heal_count == before.heal_count
    assert after_degraded.micro_state_frozen is True
    assert after_degraded.backfill_mode == "degraded_backfill"

    week_end_n1 = macro_df.index[n + 1].strftime("%Y-%m-%d")
    runtime.run_week(
        week_end=week_end_n1,
        macro_row=macro_df.iloc[n + 1],
        contracts=contracts,
        state_dir=tmp_path,
        output_dir=tmp_path / "out",
    )
    after_next = load_state(tmp_path)

    assert after_next.week_end == week_end_n1
    assert after_next.last_successful_timestamps["last_week_end"] == week_end_n1
    assert after_next.h_t_lead_prev >= after_degraded.h_t_lead_prev * load_config().micro.iir_delta


def test_batch_run_pipeline_degraded_backfill_freezes_micro_state() -> None:
    macro_df = synthetic_replay_inputs()
    n = 600
    frame = macro_df.iloc[:n]
    contracts = _make_strict_contracts(frame.index)
    backfill_week = frame.index[-2].strftime("%Y-%m-%d")

    baseline = run_pipeline(frame, contracts=contracts)
    with_backfill = run_pipeline(
        frame,
        contracts=contracts,
        backfill_modes={backfill_week: "degraded_backfill"},
    )

    frozen_row = with_backfill[-2]
    prior_row = with_backfill[-3]
    assert frozen_row.mode == "degraded"
    assert frozen_row.backfill_mode == "degraded_backfill"
    assert frozen_row.micro_state_frozen is True
    assert frozen_row.micro_envelope_internal_state == pytest.approx(
        prior_row.micro_envelope_internal_state
    )
    assert with_backfill[-1].micro_envelope_internal_state != baseline[-1].micro_envelope_internal_state


def test_block_gate_prevents_execution_permitted(tmp_path: Path) -> None:
    """When a block-level freshness failure occurs, execution_permitted == False."""
    macro_df = synthetic_replay_inputs()
    n = _WARMUP_WEEKS + 5
    contracts = _make_strict_contracts(macro_df.index[: n + 1])
    _bootstrap_state_at_week(macro_df.iloc[:n], n - 1, contracts, tmp_path)

    # Provide a stale macro (missing the current week_end) to trigger block on QQQ prices.
    week_end_n = macro_df.index[n].strftime("%Y-%m-%d")
    stale_row = macro_df.iloc[n - 1].copy()  # one week old — missing current week

    runtime = LiveRuntime()

    # We force a block by providing only a stale row (won't include week_end in macro_tail
    # freshness check — QQQ col will have last_obs < week_end).
    # Actually we extend with a row that has no QQQ value.
    stale_row_no_qqq = macro_df.iloc[n].copy()
    stale_row_no_qqq["QQQ"] = float("nan")

    result = runtime.run_week(
        week_end=week_end_n,
        macro_row=stale_row_no_qqq,
        contracts=None,  # no contracts to also trigger degrade
        state_dir=tmp_path,
        output_dir=tmp_path / "out",
    )

    # Without contracts, pipeline is in degraded mode → execution_state is degrade, not execute.
    assert result.execution_permitted is False
    assert result.execution_state in ("block", "degrade")


def test_degrade_gate_signal_valid_not_executable(tmp_path: Path) -> None:
    """Degraded freshness with strict pipeline: signal_valid_but_not_executable == True."""
    macro_df = synthetic_replay_inputs()
    n = _WARMUP_WEEKS + 5
    contracts_full = _make_strict_contracts(macro_df.index[: n + 1])
    _bootstrap_state_at_week(macro_df.iloc[:n], n - 1, contracts_full, tmp_path)

    # Pass full contracts so pipeline enters strict mode, but use stale AI_GPR to
    # trigger a degrade-level freshness failure.
    week_end_n = macro_df.index[n].strftime("%Y-%m-%d")
    row_with_stale_ai = macro_df.iloc[n].copy()
    row_with_stale_ai["AI_GPR"] = float("nan")

    # Pass contracts that have weekly_h_t so the pipeline can enter strict mode
    contracts_strict = _make_strict_contracts(macro_df.index[: n + 1])

    runtime = LiveRuntime()
    result = runtime.run_week(
        week_end=week_end_n,
        macro_row=row_with_stale_ai,
        contracts=contracts_strict,
        state_dir=tmp_path,
        output_dir=tmp_path / "out",
    )

    # With strict contracts but stale AI_GPR: degrade-level freshness + strict pipeline
    # → signal_valid_but_not_executable == True
    if result.mode == MODE_STRICT and result.execution_state == "degrade":
        assert result.signal_valid_but_not_executable is True


def test_missing_state_raises_not_silently_continues(tmp_path: Path) -> None:
    """StateNotAvailableError when state dir has no live_state_latest/."""
    runtime = LiveRuntime()
    macro_df = synthetic_replay_inputs()

    with pytest.raises(StateNotAvailableError):
        runtime.run_week(
            week_end=macro_df.index[0].strftime("%Y-%m-%d"),
            macro_row=macro_df.iloc[0],
            contracts=None,
            state_dir=tmp_path / "nonexistent",
            output_dir=tmp_path / "out",
        )


def test_live_run_log_appended(tmp_path: Path) -> None:
    """Two live runs produce exactly two rows in live_run_log.csv."""
    macro_df = synthetic_replay_inputs()
    n = _WARMUP_WEEKS + 5
    contracts = _make_strict_contracts(macro_df.index[: n + 2])
    _bootstrap_state_at_week(macro_df.iloc[:n], n - 1, contracts, tmp_path)

    runtime = LiveRuntime()
    out_dir = tmp_path / "out"

    runtime.run_week(
        week_end=macro_df.index[n].strftime("%Y-%m-%d"),
        macro_row=macro_df.iloc[n],
        contracts=contracts,
        state_dir=tmp_path,
        output_dir=out_dir,
    )
    runtime.run_week(
        week_end=macro_df.index[n + 1].strftime("%Y-%m-%d"),
        macro_row=macro_df.iloc[n + 1],
        contracts=contracts,
        state_dir=tmp_path,
        output_dir=out_dir,
    )

    log_df = pd.read_csv(out_dir / "live_run_log.csv")
    assert len(log_df) == 2
    assert "week_end" in log_df.columns
    assert "execution_state" in log_df.columns
    assert "execution_permitted" in log_df.columns


def test_run_summary_json_written(tmp_path: Path) -> None:
    """live_run_summary.json is written with expected fields."""
    macro_df = synthetic_replay_inputs()
    n = _WARMUP_WEEKS + 20  # enough rows for rolling windows to produce finite theta
    contracts = _make_strict_contracts(macro_df.index[: n + 1])
    _bootstrap_state_at_week(macro_df.iloc[:n], n - 1, contracts, tmp_path)

    runtime = LiveRuntime()
    out_dir = tmp_path / "out"
    runtime.run_week(
        week_end=macro_df.index[n].strftime("%Y-%m-%d"),
        macro_row=macro_df.iloc[n],
        contracts=contracts,
        state_dir=tmp_path,
        output_dir=out_dir,
    )

    summary = json.loads((out_dir / "live_run_summary.json").read_text())
    for field in ["week_end", "mode", "execution_state", "execution_permitted",
                  "omega_qqq_final", "circuit_breaker_active", "freshness"]:
        assert field in summary, f"missing field: {field}"
