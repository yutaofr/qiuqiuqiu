"""Phase 14 dynamic ops status summary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from qqq_cycle.config import ModelConfig, load_config
from qqq_cycle.ops.alerts import (
    SEVERITY_ORDER,
    build_alert_log,
    compute_operational_asof,
    format_operational_sla_cutoff,
    normalize_operational_now,
    required_operational_week_end,
)


@dataclass(frozen=True)
class OpsStatusArtifacts:
    """Paths written by one ops-status run."""

    summary_json_path: Path
    summary_markdown_path: Path


def build_ops_status_summary(
    *,
    latest_view: pd.DataFrame,
    revision_detail: pd.DataFrame | None = None,
    alert_log: pd.DataFrame | None = None,
    controlled_backfill_result: dict[str, Any] | None = None,
    now: pd.Timestamp | None = None,
    config: ModelConfig | None = None,
    runbook_path: str | Path = Path("docs/OPS_RUNBOOK.md"),
) -> dict[str, Any]:
    """Build the dynamic Phase 14 ops status summary."""

    model_config = config or load_config()
    operational_now = normalize_operational_now(now=now, ops_config=model_config.ops)
    required_week_end = required_operational_week_end(
        now=operational_now, ops_config=model_config.ops
    ).strftime("%Y-%m-%d")
    if alert_log is None:
        alert_log = build_alert_log(
            latest_view=latest_view,
            revision_detail=revision_detail,
            now=operational_now,
            config=model_config,
        )

    required_snapshot = _select_week(latest_view, required_week_end)
    latest_snapshot = (
        latest_view.sort_values("week_end_ts", kind="mergesort").iloc[-1]
        if not latest_view.empty
        else None
    )
    snapshot_row = required_snapshot if required_snapshot is not None else latest_snapshot
    snapshot_payload = snapshot_row["payload"] if snapshot_row is not None else {}

    signal_validity = _category_status(alert_log, "signal_validity")
    execution_readiness = _category_status(alert_log, "execution_readiness")
    data_health = _category_status(alert_log, "data_health")

    current_status = _worst_status(
        [
            signal_validity["status"],
            execution_readiness["status"],
            data_health["status"],
        ]
    )

    runbook_refs = sorted(set(alert_log["runbook_section"].tolist())) if not alert_log.empty else ["§1.1"]
    freshness_summary = _freshness_summary(snapshot_payload, required_snapshot, model_config, required_week_end)
    degraded_or_block_reasons = _collect_reasons(snapshot_payload, alert_log)
    top_reason = degraded_or_block_reasons[0] if degraded_or_block_reasons else "none"

    summary = {
        "generated_at": operational_now.tz_convert("UTC").isoformat().replace("+00:00", "Z"),
        "operational_now": operational_now.isoformat(),
        "required_week_end": required_week_end,
        "latest_available_week_end": snapshot_row["week_end"] if snapshot_row is not None else None,
        "published_at": snapshot_row["published_at_text"] if snapshot_row is not None else None,
        "operational_sla_cutoff": format_operational_sla_cutoff(model_config.ops),
        "current_mode": snapshot_row["mode"] if snapshot_row is not None else None,
        "current_status": current_status.upper(),
        "signal_validity": signal_validity,
        "execution_readiness": execution_readiness,
        "data_health": data_health,
        "freshness": freshness_summary,
        "degraded_or_block_reasons": degraded_or_block_reasons,
        "current_alerts": (
            alert_log.to_dict("records")
            if not alert_log.empty
            else []
        ),
        "runbook_path": str(runbook_path),
        "runbook_references": runbook_refs,
        "operator_action": f"see {runbook_path} {runbook_refs[0]}",
        "top_reason": top_reason,
    }
    if controlled_backfill_result is not None:
        summary["controlled_backfill"] = _controlled_backfill_status(controlled_backfill_result)
    return summary


def render_ops_status_markdown(summary: dict[str, Any]) -> str:
    """Render a human-readable ops status summary."""

    lines = [
        "# Ops Status Summary",
        "",
        f"Current status: {summary['current_status']}",
        f"Reason: {summary['top_reason']}",
        f"Operator action: {summary['operator_action']}",
        "",
        "## Snapshot",
        "",
        f"- required_week_end: {summary['required_week_end']}",
        f"- latest_available_week_end: {summary['latest_available_week_end']}",
        f"- published_at: {summary['published_at']}",
        f"- current_mode: {summary['current_mode']}",
        f"- operational_sla_cutoff: {summary['operational_sla_cutoff']}",
        "",
        "## Operational Dimensions",
        "",
        _category_line("signal_validity", summary["signal_validity"]),
        _category_line("execution_readiness", summary["execution_readiness"]),
        _category_line("data_health", summary["data_health"]),
        "",
        "## Runbook",
        "",
        f"- path: {summary['runbook_path']}",
        f"- references: {', '.join(summary['runbook_references'])}",
    ]

    if summary["current_alerts"]:
        lines.extend(["", "## Alerts", ""])
        for alert in summary["current_alerts"]:
            lines.append(
                f"- [{str(alert['alert_level']).upper()}] {alert['category']} / "
                f"{alert['alert_code']}: {alert['message']} ({alert['runbook_section']})"
            )

    return "\n".join(lines) + "\n"


def write_ops_status_outputs(
    summary: dict[str, Any],
    *,
    output_dir: str | Path,
) -> OpsStatusArtifacts:
    """Write JSON and Markdown ops status artifacts."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_json_path = out / "ops_status_summary.json"
    summary_markdown_path = out / "ops_status_summary.md"

    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_markdown_path.write_text(render_ops_status_markdown(summary), encoding="utf-8")
    return OpsStatusArtifacts(
        summary_json_path=summary_json_path,
        summary_markdown_path=summary_markdown_path,
    )


def _select_week(frame: pd.DataFrame, week_end: str) -> pd.Series | None:
    if frame.empty:
        return None
    matches = frame.loc[frame["week_end"] == week_end]
    if matches.empty:
        return None
    return matches.iloc[0]


def _category_status(alert_log: pd.DataFrame, category: str) -> dict[str, Any]:
    if alert_log.empty:
        return {"status": "ok", "reasons": []}
    category_rows = alert_log.loc[alert_log["category"] == category]
    if category_rows.empty:
        return {"status": "ok", "reasons": []}
    worst = _worst_status(category_rows["alert_level"].astype(str).tolist())
    return {
        "status": worst,
        "reasons": category_rows["message"].astype(str).tolist(),
    }


def _worst_status(statuses: list[str]) -> str:
    valid = [status for status in statuses if status in SEVERITY_ORDER]
    if not valid:
        return "ok"
    return max(valid, key=lambda status: SEVERITY_ORDER[status])


def _freshness_summary(
    payload: dict[str, Any],
    required_snapshot: pd.Series | None,
    config: ModelConfig,
    required_week_end: str,
) -> dict[str, Any]:
    freshness = payload.get("freshness", []) if payload else []
    stale_sources = [
        record.get("source_label")
        for record in freshness
        if not bool(record.get("fresh_enough"))
    ]
    published_by_sla = None
    if required_snapshot is not None:
        published_at = pd.Timestamp(required_snapshot["published_at"])
        published_by_sla = bool(
            published_at <= compute_operational_asof(required_week_end, config.ops)
        )
    return {
        "stale_sources": stale_sources,
        "stale_source_count": len(stale_sources),
        "fresh_source_count": len(freshness) - len(stale_sources),
        "published_by_sla": published_by_sla,
    }


def _collect_reasons(payload: dict[str, Any], alert_log: pd.DataFrame) -> list[str]:
    reasons: list[str] = []
    for field in ("degraded_reason", "execution_block_reason"):
        value = payload.get(field)
        if value:
            reasons.append(str(value))
    if not alert_log.empty:
        reasons.extend(alert_log["message"].astype(str).tolist())
    deduped = list(dict.fromkeys(reasons))
    return deduped


def _category_line(name: str, payload: dict[str, Any]) -> str:
    reasons = payload.get("reasons") or []
    reason_text = reasons[0] if reasons else "none"
    return f"- {name}: {str(payload['status']).upper()} ({reason_text})"


def _controlled_backfill_status(result: dict[str, Any]) -> dict[str, Any]:
    """Expose controlled-backfill mode in ops without deciding it."""

    return {
        "week_end": result.get("week_end"),
        "asset": result.get("asset"),
        "backfill_mode": result.get("backfill_mode"),
        "strict_eligible": bool(result.get("strict_eligible", False)),
        "revision_reason": result.get("revision_reason"),
        "validation_reason": result.get("validation_reason"),
        "decision_reason": result.get("decision_reason"),
    }
