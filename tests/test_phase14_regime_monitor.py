"""Tests for Phase 14 latest-view regime monitoring."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qqq_cycle.ops.publishing import publish_from_live_summary_path
from qqq_cycle.ops.regime_monitor import (
    build_event_response_summary,
    build_state_duration_summary,
    build_state_transition_matrix,
    load_latest_snapshot_per_week,
)


def _write_live_summary(path: Path, *, week_end: str, mode: str = "strict", k_hat_t: int = 2, s_t: float = 0.31, h_t: float = 0.42, rho_t: float = 0.28, drift_flag: int = 1) -> None:
    summary = {
        "run_timestamp": "2026-04-29T08:00:00Z",
        "week_end": week_end,
        "mode": mode,
        "execution_state": "execute" if mode == "strict" else "degrade",
        "execution_permitted": mode == "strict",
        "signal_valid_but_not_executable": mode != "strict",
        "backfill_mode": None,
        "micro_state_frozen": False,
        "micro_envelope_internal_state": h_t,
        "micro_breaker_internal_state": "inactive",
        "micro_rho_update_state": "observed",
        "contract_source": "stores_strict",
        "strict_gate_passed": True,
        "degraded_reason": None if mode == "strict" else "strict contracts not satisfied",
        "execution_block_reason": None,
        "strict_contracts_satisfied": mode == "strict",
        "k_hat_t": k_hat_t,
        "p_t": [0.05, 0.10, 0.70, 0.10, 0.05],
        "s_t": s_t,
        "h_t": h_t,
        "rho_t": rho_t,
        "I_t": {
            "A_t": {"H_components": [0.1, 0.2, 0.3]},
            "C_t": {"c_rule": 0},
            "D_t": {"d_state": 0.2},
            "H_t": {"h_macro": 1},
        },
        "interpretability": {
            "H": 0.45,
            "I": -0.10,
            "L": 0.20,
            "T": 0.30,
            "P": 0.15,
            "E": -0.05,
            "drift_flag": drift_flag,
        },
        "omega_qqq_final": 0.75,
        "omega_shy_final": 0.25,
        "circuit_breaker_active": False,
        "rebalance_required": False,
        "reason": "ok",
        "freshness": [
            {
                "source_label": "fred_macro",
                "last_observation_date": week_end,
                "fresh_enough": True,
                "blocking_level": "degrade",
                "reason": None,
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _publish(
    tmp_path: Path,
    *,
    week_end: str,
    published_at: str,
    mode: str = "strict",
    k_hat_t: int = 2,
    s_t: float = 0.31,
    h_t: float = 0.42,
    rho_t: float = 0.28,
    drift_flag: int = 1,
) -> Path:
    summary_path = tmp_path / "live" / f"{week_end}_{published_at.replace(':', '-')}.json"
    _write_live_summary(
        summary_path,
        week_end=week_end,
        mode=mode,
        k_hat_t=k_hat_t,
        s_t=s_t,
        h_t=h_t,
        rho_t=rho_t,
        drift_flag=drift_flag,
    )
    output_dir = tmp_path / "phase14"
    publish_from_live_summary_path(
        summary_path=summary_path,
        output_dir=output_dir,
        published_at=published_at,
    )
    return output_dir / "history"


def test_latest_snapshot_per_week_loader(tmp_path: Path) -> None:
    history_dir = _publish(
        tmp_path,
        week_end="2026-04-24",
        published_at="2026-04-29T10:00:00Z",
        k_hat_t=1,
    )
    _publish(
        tmp_path,
        week_end="2026-04-24",
        published_at="2026-04-29T10:05:00Z",
        k_hat_t=3,
    )
    _publish(
        tmp_path,
        week_end="2026-05-01",
        published_at="2026-05-06T10:00:00Z",
        k_hat_t=2,
    )

    latest = load_latest_snapshot_per_week(history_dir)
    assert list(latest["week_end"]) == ["2026-04-24", "2026-05-01"]
    first = latest.iloc[0]
    assert first["published_at_text"] == "2026-04-29T10:05:00Z"
    assert first["k_hat_t"] == 3


def test_regime_monitor_uses_latest_view_only(tmp_path: Path) -> None:
    history_dir = _publish(
        tmp_path,
        week_end="2026-04-24",
        published_at="2026-04-29T10:00:00Z",
        k_hat_t=1,
    )
    _publish(
        tmp_path,
        week_end="2026-04-24",
        published_at="2026-04-29T10:05:00Z",
        k_hat_t=4,
    )
    _publish(
        tmp_path,
        week_end="2026-05-01",
        published_at="2026-05-06T10:00:00Z",
        k_hat_t=2,
    )

    latest = load_latest_snapshot_per_week(history_dir)
    matrix = build_state_transition_matrix(latest)

    assert "strict:k1" not in matrix.index
    assert matrix.loc["strict:k4", "strict:k2"] == 1


def test_transition_matrix_and_duration_outputs_valid(tmp_path: Path) -> None:
    history_dir = _publish(
        tmp_path,
        week_end="2026-04-24",
        published_at="2026-04-29T10:00:00Z",
        k_hat_t=1,
        s_t=0.10,
    )
    _publish(
        tmp_path,
        week_end="2026-05-01",
        published_at="2026-05-06T10:00:00Z",
        k_hat_t=1,
        s_t=0.12,
    )
    _publish(
        tmp_path,
        week_end="2026-05-08",
        published_at="2026-05-13T10:00:00Z",
        k_hat_t=3,
        s_t=0.40,
        h_t=0.55,
        rho_t=0.60,
    )

    latest = load_latest_snapshot_per_week(history_dir)
    matrix = build_state_transition_matrix(latest)
    duration = build_state_duration_summary(latest)
    events = build_event_response_summary(latest)

    assert int(matrix.to_numpy().sum()) == 2
    run_row = duration.loc[duration["regime_label"] == "strict:k1"].iloc[0]
    assert int(run_row["max_duration_weeks"]) == 2
    assert len(events) == 1
    assert events.iloc[0]["from_regime"] == "strict:k1"
    assert events.iloc[0]["to_regime"] == "strict:k3"
