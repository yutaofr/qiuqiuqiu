"""Tests for Phase 12B interpretability snapshot and dashboard CSV exports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qqq_cycle.backtest.diagnostics import synthetic_replay_inputs
from qqq_cycle.config import load_config
from qqq_cycle.core.covariance import RobustEWCov2D
from qqq_cycle.core.proto_online import initialize_prototypes_from_history
from qqq_cycle.live.dashboard import (
    append_drift_monitor,
    append_pollution_flags,
    append_state_plane,
)
from qqq_cycle.live.freshness import FreshnessRecord
from qqq_cycle.live.interpretability import (
    InterpretabilitySnapshot,
    build_snapshot,
    snapshot_to_dict,
)
from qqq_cycle.live.runtime import LiveRuntime
from qqq_cycle.live.state_io import LiveState, save_state
from qqq_cycle.pipeline import MODE_STRICT, MODE_DEGRADED, PipelineContracts, PipelineResult

_WARMUP_WEEKS = 260


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-03", periods=n, freq="W-FRI")


def _make_macro_tail(n: int = 10) -> pd.DataFrame:
    dates = _make_dates(n)
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.standard_normal((n, 8)),
        index=dates,
        columns=["DFII10", "DGS2", "BAMLH0A0HYM2", "NFCI", "VIXCLS", "AI_GPR", "USEPUINDXD", "QQQ"],
    )


def _make_cov_state():
    rng = np.random.default_rng(42)
    hist = rng.standard_normal((25, 2))
    cov = RobustEWCov2D()
    state = cov.initialize_from_history(hist)
    for row in hist[20:]:
        state = cov.update(state, row)
    return state


def _make_proto():
    rng = np.random.default_rng(42)
    hist = rng.standard_normal((50, 2))
    return initialize_prototypes_from_history(hist)


def _make_live_state(with_proto: bool = True) -> LiveState:
    cov_state = _make_cov_state()
    proto = _make_proto() if with_proto else None
    return LiveState(
        week_end="2024-01-05",
        cov_state=cov_state,
        proto=proto,
        proto_seed=[],
        h_t_lead_prev=0.35,
        heal_count=1,
        warmup_count=cov_state.warmup_count,
        breaker_active=False,
        weeks_outside_s1=0,
        prev_omega_qqq=0.60,
        macro_tail=_make_macro_tail(),
        last_successful_timestamps={},
    )


def _make_pipeline_result(mode: str = MODE_STRICT) -> PipelineResult:
    return PipelineResult(
        week_end="2024-01-05",
        k_hat_t=2,
        p_t=[0.10, 0.15, 0.50, 0.15, 0.10],
        s_t=0.42,
        h_t=0.30,
        rho_t=0.55,
        I_t=0.12,
        interpretability={
            "L": 0.8, "T": 0.6, "P": 0.5, "E": 0.3, "H": 1.1, "I": 0.12,
            "d": 0.05, "a": 0.02, "g_raw": 0.07, "g_stress": 0.06, "s": 0.42,
            "drift_probe_raw": 0.9, "drift_flag": 0,
            "k_hat_t": 2, "p_t": [0.10, 0.15, 0.50, 0.15, 0.10],
            "h_t": 0.30, "rho_t": 0.55,
        },
        mode=mode,
        degraded_reason=None,
        strict_contracts_satisfied=True,
    )


def _make_freshness(all_fresh: bool = True) -> list[FreshnessRecord]:
    sources = [
        ("fred_macro", "degrade"),
        ("ai_gpr", "degrade"),
        ("qqq_prices", "block"),
        ("constituents", "block"),
        ("weights", "block"),
        ("pit_engine", "block"),
    ]
    return [
        FreshnessRecord(
            source_label=label,
            last_observation_date="2024-01-05",
            asof_timestamp="2024-01-05",
            fresh_enough=all_fresh,
            blocking_level=level,
            reason=None if all_fresh else f"{label} stale",
        )
        for label, level in sources
    ]


# ---------------------------------------------------------------------------
# Test 1: snapshot schema
# ---------------------------------------------------------------------------

def test_snapshot_has_all_required_fields() -> None:
    """build_snapshot() returns an InterpretabilitySnapshot with all expected fields."""
    config = load_config()
    snap = build_snapshot(
        week_end="2024-01-05",
        pipeline_result=_make_pipeline_result(),
        freshness=_make_freshness(all_fresh=True),
        execution_state="execute",
        execution_permitted=True,
        signal_valid_but_not_executable=False,
        live_state=_make_live_state(with_proto=True),
        config=config,
    )
    assert isinstance(snap, InterpretabilitySnapshot)
    assert snap.week_end == "2024-01-05"
    assert snap.execution_tier == "signal_valid_executable"
    assert snap.H_t is not None
    assert snap.I_t is not None
    assert snap.k_hat_t == 2
    assert snap.state_label is not None
    assert snap.state_probabilities is not None
    assert len(snap.state_probabilities) == 5
    assert snap.centroids is not None
    assert len(snap.centroids) == 5
    assert all(len(c) == 2 for c in snap.centroids)
    for key in ("L", "T", "P", "E"):
        assert key in snap.factor_attribution
    for key in ("displacement_d", "acceleration_a", "g_raw", "g_stress", "s_t"):
        assert key in snap.stress_attribution
    for key in ("drift_probe_raw", "drift_flag", "threshold_lo", "threshold_hi"):
        assert key in snap.drift_metrics
    for key in ("warmup_count", "heal_count", "breaker_active", "h_t_lead_prev"):
        assert key in snap.health_metrics


# ---------------------------------------------------------------------------
# Test 2: execution_tier routing
# ---------------------------------------------------------------------------

def test_execution_tier_all_paths() -> None:
    """execution_tier is set correctly for all three routing outcomes."""
    config = load_config()
    live_state = _make_live_state()
    freshness = _make_freshness(all_fresh=True)
    pr = _make_pipeline_result()

    # execute path
    snap = build_snapshot(
        week_end="2024-01-05", pipeline_result=pr, freshness=freshness,
        execution_state="execute", execution_permitted=True,
        signal_valid_but_not_executable=False, live_state=live_state, config=config,
    )
    assert snap.execution_tier == "signal_valid_executable"

    # degrade + strict → signal_valid_not_executable
    snap = build_snapshot(
        week_end="2024-01-05", pipeline_result=pr, freshness=freshness,
        execution_state="degrade", execution_permitted=False,
        signal_valid_but_not_executable=True, live_state=live_state, config=config,
    )
    assert snap.execution_tier == "signal_valid_not_executable"

    # block → signal_invalid
    snap = build_snapshot(
        week_end="2024-01-05", pipeline_result=pr, freshness=freshness,
        execution_state="block", execution_permitted=False,
        signal_valid_but_not_executable=False, live_state=live_state, config=config,
    )
    assert snap.execution_tier == "signal_invalid"


# ---------------------------------------------------------------------------
# Test 3: pollution_flags reflect freshness records
# ---------------------------------------------------------------------------

def test_pollution_flags_stale_sources() -> None:
    """Stale freshness records are reflected in pollution_flags.stale_sources."""
    config = load_config()
    live_state = _make_live_state()
    pr = _make_pipeline_result()
    freshness = _make_freshness(all_fresh=False)

    snap = build_snapshot(
        week_end="2024-01-05", pipeline_result=pr, freshness=freshness,
        execution_state="block", execution_permitted=False,
        signal_valid_but_not_executable=False, live_state=live_state, config=config,
    )
    pf = snap.pollution_flags
    assert len(pf["stale_sources"]) == len(freshness)
    for rec in freshness:
        assert pf.get(f"{rec.source_label}_fresh") is False


def test_pollution_flags_all_fresh() -> None:
    """When all sources are fresh, stale_sources is empty."""
    config = load_config()
    snap = build_snapshot(
        week_end="2024-01-05", pipeline_result=_make_pipeline_result(),
        freshness=_make_freshness(all_fresh=True),
        execution_state="execute", execution_permitted=True,
        signal_valid_but_not_executable=False, live_state=_make_live_state(), config=config,
    )
    assert snap.pollution_flags["stale_sources"] == []


# ---------------------------------------------------------------------------
# Test 4: snapshot_to_dict is JSON-serializable
# ---------------------------------------------------------------------------

def test_snapshot_to_dict_json_round_trip() -> None:
    """snapshot_to_dict() produces a JSON-serializable dict with required keys."""
    config = load_config()
    snap = build_snapshot(
        week_end="2024-01-05", pipeline_result=_make_pipeline_result(),
        freshness=_make_freshness(all_fresh=True),
        execution_state="execute", execution_permitted=True,
        signal_valid_but_not_executable=False, live_state=_make_live_state(), config=config,
    )
    d = snapshot_to_dict(snap)
    serialized = json.dumps(d)  # must not raise
    restored = json.loads(serialized)
    for key in ("week_end", "execution_tier", "state_plane", "factor_attribution",
                "stress_attribution", "drift_metrics", "health_metrics", "pollution_flags"):
        assert key in restored


# ---------------------------------------------------------------------------
# Test 5: dashboard CSVs are appended on each live run
# ---------------------------------------------------------------------------

def test_dashboard_csvs_appended_on_live_runs(tmp_path: Path) -> None:
    """Two live runs produce two rows in each dashboard CSV."""
    macro_df = synthetic_replay_inputs()
    n = _WARMUP_WEEKS + 20
    contracts = _make_strict_contracts(macro_df.index[: n + 2])
    _bootstrap(macro_df.iloc[:n], n - 1, contracts, tmp_path)

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

    for fname in ("dashboard_state_plane.csv", "dashboard_drift_monitor.csv",
                  "dashboard_pollution_flags.csv"):
        df = pd.read_csv(out_dir / fname)
        assert len(df) == 2, f"{fname}: expected 2 rows, got {len(df)}"
        assert "week_end" in df.columns


# ---------------------------------------------------------------------------
# Test 6: interpretability_snapshot_latest.json written with correct fields
# ---------------------------------------------------------------------------

def test_interpretability_snapshot_json_written(tmp_path: Path) -> None:
    """interpretability_snapshot_latest.json is written after each live run."""
    macro_df = synthetic_replay_inputs()
    n = _WARMUP_WEEKS + 20
    contracts = _make_strict_contracts(macro_df.index[: n + 1])
    _bootstrap(macro_df.iloc[:n], n - 1, contracts, tmp_path)

    runtime = LiveRuntime()
    out_dir = tmp_path / "out"
    runtime.run_week(
        week_end=macro_df.index[n].strftime("%Y-%m-%d"),
        macro_row=macro_df.iloc[n],
        contracts=contracts,
        state_dir=tmp_path,
        output_dir=out_dir,
    )

    snap_path = out_dir / "interpretability_snapshot_latest.json"
    assert snap_path.exists(), "interpretability_snapshot_latest.json not written"
    data = json.loads(snap_path.read_text())
    for key in ("week_end", "execution_tier", "state_plane", "factor_attribution",
                "stress_attribution", "drift_metrics", "health_metrics", "pollution_flags"):
        assert key in data, f"missing key: {key}"
    assert data["pollution_flags"]["execution_tier"] in (
        "signal_valid_executable", "signal_valid_not_executable", "signal_invalid"
    )


# ---------------------------------------------------------------------------
# Helpers reused from test_live_runtime (duplicated to keep tests self-contained)
# ---------------------------------------------------------------------------

from qqq_cycle.pipeline import _check_strict_gate, _safe_float
from qqq_cycle.live.runtime import _run_pipeline_step
from qqq_cycle.core.covariance import RobustEWCov2D as _RobustEWCov2D
from qqq_cycle.core.drift_probe import DriftProbe
from qqq_cycle.core.state_layer import compute_state_layer
from qqq_cycle.core.stress_layer import compute_stress_layer


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


def _bootstrap(
    macro_df: pd.DataFrame,
    up_to_idx: int,
    contracts: PipelineContracts,
    state_dir: Path,
) -> None:
    config = load_config()
    tail = macro_df.iloc[: up_to_idx + 1]
    state_frame = compute_state_layer(tail)
    theta = state_frame[["H", "I"]]
    stress_result = compute_stress_layer(theta, state_frame["E"])
    stress_frame = stress_result.frame
    drift_frame = DriftProbe(
        theta_lo=config.drift.theta_lo, theta_hi=config.drift.theta_hi,
    ).compute(tail)

    finite_theta = theta.dropna()
    cov = _RobustEWCov2D(warmup_weeks=config.warmup_weeks)
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
        h_t_raw = None
        if can_h and contracts.weekly_h_t is not None and week_end in contracts.weekly_h_t.index:
            h_t_raw = _safe_float(contracts.weekly_h_t.loc[week_end])

        _, cov_state, proto, proto_seed, h_t_lead_prev, heal_count = _run_pipeline_step(
            week_end=week_end, theta_row=x, s_t_val=s_t, i_t_val=i_t,
            drift_raw=drift_raw, drift_flag=drift_flag, h_t_raw=h_t_raw,
            cov_state=cov_state, proto=proto, proto_seed=proto_seed,
            h_t_lead_prev=h_t_lead_prev, heal_count=heal_count, config=config,
            can_compute_h_t=can_h, degraded_reason=deg_reason, week_index=idx,
            state_frame=state_frame, stress_frame=stress_frame, drift_frame=drift_frame,
        )

    live_state = LiveState(
        week_end=tail.index[-1].strftime("%Y-%m-%d"),
        cov_state=cov_state, proto=proto, proto_seed=proto_seed,
        h_t_lead_prev=h_t_lead_prev, heal_count=heal_count,
        warmup_count=cov_state.warmup_count, breaker_active=False,
        weeks_outside_s1=0, prev_omega_qqq=0.5, macro_tail=tail,
        last_successful_timestamps={},
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    save_state(live_state, state_dir)
