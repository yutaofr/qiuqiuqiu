"""Tests for Phase 14 SLA-based alerts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qqq_cycle.config import load_config
from qqq_cycle.ops.alerts import build_alert_log
from qqq_cycle.ops.publishing import publish_from_live_summary_path
from qqq_cycle.ops.regime_monitor import load_latest_snapshot_per_week
from qqq_cycle.ops.revision_audit import build_revision_detail, load_snapshot_history


def _write_live_summary(
    path: Path,
    *,
    week_end: str,
    mode: str = "strict",
    execution_state: str = "execute",
    k_hat_t: int = 2,
    s_t: float = 0.31,
    h_t: float = 0.42,
    rho_t: float = 0.28,
    freshness: list[dict] | None = None,
) -> None:
    summary = {
        "run_timestamp": "2026-04-29T08:00:00Z",
        "week_end": week_end,
        "mode": mode,
        "execution_state": execution_state,
        "execution_permitted": execution_state == "execute",
        "signal_valid_but_not_executable": execution_state == "degrade",
        "backfill_mode": None,
        "micro_state_frozen": False,
        "micro_envelope_internal_state": h_t,
        "micro_breaker_internal_state": "inactive",
        "micro_rho_update_state": "observed",
        "contract_source": "stores_strict",
        "strict_gate_passed": True,
        "degraded_reason": None if execution_state == "execute" else "degraded execution",
        "execution_block_reason": None if execution_state != "block" else "blocked execution",
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
            "drift_flag": 1,
        },
        "omega_qqq_final": 0.75,
        "omega_shy_final": 0.25,
        "circuit_breaker_active": False,
        "rebalance_required": False,
        "reason": "ok",
        "freshness": freshness
        or [
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
    execution_state: str = "execute",
    k_hat_t: int = 2,
    s_t: float = 0.31,
    h_t: float = 0.42,
    rho_t: float = 0.28,
    freshness: list[dict] | None = None,
) -> Path:
    summary_path = tmp_path / "live" / f"{week_end}_{published_at.replace(':', '-')}.json"
    _write_live_summary(
        summary_path,
        week_end=week_end,
        mode=mode,
        execution_state=execution_state,
        k_hat_t=k_hat_t,
        s_t=s_t,
        h_t=h_t,
        rho_t=rho_t,
        freshness=freshness,
    )
    output_dir = tmp_path / "phase14"
    publish_from_live_summary_path(
        summary_path=summary_path,
        output_dir=output_dir,
        published_at=published_at,
    )
    return output_dir / "history"


def test_staleness_is_evaluated_against_operational_sla_cutoff(tmp_path: Path) -> None:
    history_dir = _publish(
        tmp_path,
        week_end="2026-04-17",
        published_at="2026-04-18T15:00:00Z",
    )
    latest = load_latest_snapshot_per_week(history_dir)
    revision_detail = build_revision_detail(load_snapshot_history(history_dir))
    config = load_config()

    before_cutoff = build_alert_log(
        latest_view=latest,
        revision_detail=revision_detail,
        now=pd.Timestamp("2026-04-25T11:00:00-04:00"),
        config=config,
    )
    after_cutoff = build_alert_log(
        latest_view=latest,
        revision_detail=revision_detail,
        now=pd.Timestamp("2026-04-25T12:30:00-04:00"),
        config=config,
    )

    assert "missing_required_snapshot" not in before_cutoff["alert_code"].tolist()
    assert "missing_required_snapshot" in after_cutoff["alert_code"].tolist()


def test_warn_degrade_block_alert_generation(tmp_path: Path) -> None:
    history_dir = _publish(
        tmp_path,
        week_end="2026-04-24",
        published_at="2026-04-26T18:00:00Z",
        execution_state="degrade",
        freshness=[
            {
                "source_label": "fred_macro",
                "last_observation_date": "2026-04-17",
                "fresh_enough": False,
                "blocking_level": "degrade",
                "reason": "macro lag",
            },
            {
                "source_label": "qqq_prices",
                "last_observation_date": "2026-04-17",
                "fresh_enough": False,
                "blocking_level": "block",
                "reason": "price lag",
            },
        ],
    )
    latest = load_latest_snapshot_per_week(history_dir)
    revision_detail = build_revision_detail(load_snapshot_history(history_dir))
    alerts = build_alert_log(
        latest_view=latest,
        revision_detail=revision_detail,
        now=pd.Timestamp("2026-04-27T12:00:00-04:00"),
        config=load_config(),
    )

    levels = set(alerts["alert_level"].tolist())
    assert {"warn", "degrade", "block"}.issubset(levels)
