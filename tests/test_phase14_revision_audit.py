"""Tests for Phase 14 immutable-history revision audit."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from qqq_cycle.ops.publishing import publish_from_live_summary_path
from qqq_cycle.ops.revision_audit import (
    MATERIAL_DELTA_THRESHOLD,
    build_revision_detail,
    build_revision_summary,
    build_revision_tests,
    load_snapshot_history,
)


def _write_live_summary(path: Path, *, week_end: str, mode: str = "strict", k_hat_t: int = 2, s_t: float = 0.31, h_t: float = 0.42, rho_t: float = 0.28, p_t: list[float] | None = None, drift_flag: int = 1) -> None:
    summary = {
        "run_timestamp": "2026-04-29T08:00:00Z",
        "week_end": week_end,
        "mode": mode,
        "execution_state": "execute",
        "execution_permitted": True,
        "signal_valid_but_not_executable": False,
        "degraded_reason": None,
        "execution_block_reason": None,
        "strict_contracts_satisfied": True,
        "k_hat_t": k_hat_t,
        "p_t": p_t or [0.05, 0.10, 0.70, 0.10, 0.05],
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
    p_t: list[float] | None = None,
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
        p_t=p_t,
        drift_flag=drift_flag,
    )
    output_dir = tmp_path / "phase14"
    publish_from_live_summary_path(
        summary_path=summary_path,
        output_dir=output_dir,
        published_at=published_at,
    )
    return output_dir / "history"


def test_revision_audit_uses_earliest_and_latest_per_week(tmp_path: Path) -> None:
    history_dir = _publish(
        tmp_path,
        week_end="2026-04-24",
        published_at="2026-04-29T10:05:00Z",
        s_t=0.20,
    )
    _publish(
        tmp_path,
        week_end="2026-04-24",
        published_at="2026-04-29T10:00:00Z",
        s_t=0.10,
    )

    history = load_snapshot_history(history_dir)
    detail = build_revision_detail(history)

    row = detail.iloc[0]
    assert row["week_end"] == "2026-04-24"
    assert row["first_published_at"] == "2026-04-29T10:00:00Z"
    assert row["latest_published_at"] == "2026-04-29T10:05:00Z"
    assert row["delta_s"] == 0.10


def test_revision_audit_survives_multiple_same_week_runs(tmp_path: Path) -> None:
    history_dir = _publish(
        tmp_path,
        week_end="2026-04-24",
        published_at="2026-04-29T10:00:00Z",
    )
    _publish(
        tmp_path,
        week_end="2026-04-24",
        published_at="2026-04-29T10:05:00Z",
        p_t=[0.10, 0.10, 0.60, 0.10, 0.10],
    )
    _publish(
        tmp_path,
        week_end="2026-04-24",
        published_at="2026-04-29T10:10:00Z",
        drift_flag=0,
    )
    _publish(
        tmp_path,
        week_end="2026-05-01",
        published_at="2026-05-06T10:00:00Z",
        mode="degraded",
        k_hat_t=1,
    )

    history = load_snapshot_history(history_dir)
    detail = build_revision_detail(history)
    summary = build_revision_summary(detail)
    tests_payload = build_revision_tests(detail)

    row = detail.loc[detail["week_end"] == "2026-04-24"].iloc[0]
    assert row["run_count"] == 3
    assert bool(row["drift_flag_changed"]) is True
    assert int(summary.iloc[0]["weeks_with_multiple_runs"]) == 1
    assert tests_payload["checks"]["same_week_multiple_runs_supported"] is True


def test_material_revision_thresholds_applied_correctly(tmp_path: Path) -> None:
    history_dir = _publish(
        tmp_path,
        week_end="2026-04-24",
        published_at="2026-04-29T10:00:00Z",
        s_t=0.10,
        h_t=0.20,
        rho_t=0.30,
        k_hat_t=2,
    )
    _publish(
        tmp_path,
        week_end="2026-04-24",
        published_at="2026-04-29T10:05:00Z",
        s_t=0.10 + MATERIAL_DELTA_THRESHOLD,
        h_t=0.20 + 0.06,
        rho_t=0.30 + 0.04,
        k_hat_t=2,
    )
    _publish(
        tmp_path,
        week_end="2026-05-01",
        published_at="2026-05-06T10:00:00Z",
        mode="strict",
        k_hat_t=1,
    )
    _publish(
        tmp_path,
        week_end="2026-05-01",
        published_at="2026-05-06T10:05:00Z",
        mode="degraded",
        k_hat_t=1,
    )

    history = load_snapshot_history(history_dir)
    detail = build_revision_detail(history)

    delta_row = detail.loc[detail["week_end"] == "2026-04-24"].iloc[0]
    mode_row = detail.loc[detail["week_end"] == "2026-05-01"].iloc[0]

    assert pd.isna(delta_row["delta_s"]) is False
    assert math.isclose(abs(float(delta_row["delta_s"])), MATERIAL_DELTA_THRESHOLD, rel_tol=0.0, abs_tol=1e-12)
    assert bool(delta_row["material_revision"]) is True
    assert "delta_h_gt_0.05" in str(delta_row["revision_reason"])
    assert "delta_s_gt_0.05" not in str(delta_row["revision_reason"])

    assert bool(mode_row["material_revision"]) is True
    assert "mode_changed" in str(mode_row["revision_reason"])
