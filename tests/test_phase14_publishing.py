"""Tests for Phase 14 immutable publishing."""

from __future__ import annotations

import json
from pathlib import Path

from qqq_cycle.ops.publishing import publish_from_live_summary_path


def _write_live_summary(path: Path) -> dict:
    summary = {
        "run_timestamp": "2026-04-29T08:00:00Z",
        "week_end": "2026-04-24",
        "mode": "strict",
        "execution_state": "execute",
        "execution_permitted": True,
        "signal_valid_but_not_executable": False,
        "degraded_reason": None,
        "execution_block_reason": None,
        "strict_contracts_satisfied": True,
        "k_hat_t": 2,
        "p_t": [0.05, 0.10, 0.70, 0.10, 0.05],
        "s_t": 0.31,
        "h_t": 0.42,
        "rho_t": 0.28,
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
            "drift_flag": 1,
        },
        "omega_qqq_final": 0.75,
        "omega_shy_final": 0.25,
        "circuit_breaker_active": False,
        "rebalance_required": False,
        "reason": "ok",
        "freshness": [
            {
                "source_label": "fred_macro",
                "last_observation_date": "2026-04-24",
                "fresh_enough": True,
                "blocking_level": "degrade",
                "reason": None,
            },
            {
                "source_label": "qqq_prices",
                "last_observation_date": "2026-04-24",
                "fresh_enough": True,
                "blocking_level": "block",
                "reason": None,
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_history_snapshots_are_immutable(tmp_path: Path) -> None:
    summary_path = tmp_path / "live" / "live_run_summary.json"
    _write_live_summary(summary_path)
    output_dir = tmp_path / "phase14"

    _, first_artifacts = publish_from_live_summary_path(
        summary_path=summary_path,
        output_dir=output_dir,
        published_at="2026-04-29T10:00:00Z",
    )
    first_contents = first_artifacts.snapshot_history_path.read_text(encoding="utf-8")

    _, second_artifacts = publish_from_live_summary_path(
        summary_path=summary_path,
        output_dir=output_dir,
        published_at="2026-04-29T10:05:00Z",
    )

    assert first_artifacts.snapshot_history_path != second_artifacts.snapshot_history_path
    assert first_artifacts.snapshot_history_path.read_text(encoding="utf-8") == first_contents


def test_same_week_multiple_runs_create_multiple_files(tmp_path: Path) -> None:
    summary_path = tmp_path / "live" / "live_run_summary.json"
    _write_live_summary(summary_path)
    output_dir = tmp_path / "phase14"

    publish_from_live_summary_path(
        summary_path=summary_path,
        output_dir=output_dir,
        published_at="2026-04-29T10:00:00Z",
    )
    publish_from_live_summary_path(
        summary_path=summary_path,
        output_dir=output_dir,
        published_at="2026-04-29T10:05:00Z",
    )

    history_snapshot_files = sorted((output_dir / "history").glob("cycle_snapshot_*.json"))
    history_report_files = sorted(output_dir.glob("weekly_cycle_report_*__run_*.md"))
    assert len(history_snapshot_files) == 2
    assert len(history_report_files) == 2


def test_latest_snapshot_points_to_latest_run(tmp_path: Path) -> None:
    summary_path = tmp_path / "live" / "live_run_summary.json"
    _write_live_summary(summary_path)
    output_dir = tmp_path / "phase14"

    publish_from_live_summary_path(
        summary_path=summary_path,
        output_dir=output_dir,
        published_at="2026-04-29T10:00:00Z",
    )
    snapshot, artifacts = publish_from_live_summary_path(
        summary_path=summary_path,
        output_dir=output_dir,
        published_at="2026-04-29T10:05:00Z",
    )

    latest_snapshot = _load_json(artifacts.snapshot_latest_path)
    latest_history_snapshot = _load_json(artifacts.snapshot_history_path)
    latest_report = artifacts.report_latest_path.read_text(encoding="utf-8")
    latest_history_report = artifacts.report_history_path.read_text(encoding="utf-8")

    assert latest_snapshot == latest_history_snapshot
    assert latest_snapshot["published_at"] == snapshot.published_at
    assert latest_report == latest_history_report


def test_weekly_report_matches_latest_snapshot(tmp_path: Path) -> None:
    summary_path = tmp_path / "live" / "live_run_summary.json"
    source_summary = _write_live_summary(summary_path)
    output_dir = tmp_path / "phase14"

    snapshot, artifacts = publish_from_live_summary_path(
        summary_path=summary_path,
        output_dir=output_dir,
        published_at="2026-04-29T10:05:00Z",
    )

    latest_snapshot = _load_json(artifacts.snapshot_latest_path)
    latest_report = artifacts.report_latest_path.read_text(encoding="utf-8")

    assert latest_snapshot["week_end"] == source_summary["week_end"]
    assert latest_snapshot["mode"] == source_summary["mode"]
    assert latest_snapshot["p_t"] == source_summary["p_t"]
    assert latest_snapshot["drift_flag"] == source_summary["interpretability"]["drift_flag"]

    assert f"- week_end: {snapshot.week_end}" in latest_report
    assert f"- published_at: {snapshot.published_at}" in latest_report
    assert f"- source_hash: {snapshot.source_hash}" in latest_report
    assert f"- mode: {snapshot.mode}" in latest_report
    assert f"- s_t: {snapshot.s_t}" in latest_report
    assert f"- h_t: {snapshot.h_t}" in latest_report
    assert f"- rho_t: {snapshot.rho_t}" in latest_report
    assert f"- drift_flag: {snapshot.drift_flag}" in latest_report
