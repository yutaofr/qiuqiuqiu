"""Phase 14 SLA-based operational alerts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from qqq_cycle.config import ModelConfig, OpsConfig, load_config

SEVERITY_ORDER = {"ok": 0, "warn": 1, "degrade": 2, "block": 3}
_WEEK_END_WEEKDAY = 4  # Friday
_WEEKDAY_INDEX = {
    "MON": 0,
    "TUE": 1,
    "WED": 2,
    "THU": 3,
    "FRI": 4,
    "SAT": 5,
    "SUN": 6,
}

_RUNBOOK_SECTION_BY_CODE = {
    "missing_required_snapshot": "§2.1",
    "late_snapshot_publication": "§2.2",
    "stale_source_warn": "§2.3",
    "stale_source_degrade": "§2.3",
    "stale_source_block": "§2.3",
    "execution_degraded": "§3.1",
    "execution_blocked": "§3.2",
    "signal_invalid": "§3.3",
    "material_revision_latest_week": "§4.1",
}


@dataclass(frozen=True)
class AlertArtifacts:
    """Paths written by one operational-alert run."""

    alert_log_path: Path


def format_operational_sla_cutoff(ops_config: OpsConfig) -> str:
    """Render the configured SLA cutoff as a human-readable label."""

    return (
        f"{ops_config.sla_cutoff_weekday} "
        f"{ops_config.sla_cutoff_time} "
        f"{ops_config.operational_timezone}"
    )


def compute_operational_asof(
    week_end: str | pd.Timestamp,
    ops_config: OpsConfig,
) -> pd.Timestamp:
    """Return the SLA cutoff timestamp for a decision week_end."""

    tz = ZoneInfo(ops_config.operational_timezone)
    week_end_ts = pd.Timestamp(week_end).normalize()
    cutoff_weekday = _WEEKDAY_INDEX[ops_config.sla_cutoff_weekday.upper()]
    day_offset = (cutoff_weekday - week_end_ts.weekday()) % 7
    cutoff_date = week_end_ts + pd.Timedelta(days=day_offset)
    hour_text, minute_text = ops_config.sla_cutoff_time.split(":", 1)
    return pd.Timestamp(
        year=cutoff_date.year,
        month=cutoff_date.month,
        day=cutoff_date.day,
        hour=int(hour_text),
        minute=int(minute_text),
        tz=tz,
    )


def required_operational_week_end(
    *,
    now: pd.Timestamp | None = None,
    ops_config: OpsConfig,
) -> pd.Timestamp:
    """Return the latest week_end that should be available by the SLA cutoff."""

    now_local = normalize_operational_now(now=now, ops_config=ops_config)
    candidate = now_local.normalize() - pd.Timedelta(
        days=(now_local.weekday() - _WEEK_END_WEEKDAY) % 7
    )
    cutoff = compute_operational_asof(candidate, ops_config)
    if now_local >= cutoff:
        return candidate.tz_localize(None)
    return (candidate - pd.Timedelta(days=7)).tz_localize(None)


def normalize_operational_now(
    *,
    now: pd.Timestamp | None,
    ops_config: OpsConfig,
) -> pd.Timestamp:
    """Normalize wall-clock time into the configured operational timezone."""

    tz = ZoneInfo(ops_config.operational_timezone)
    if now is None:
        return pd.Timestamp.now(tz=tz)
    ts = pd.Timestamp(now)
    if ts.tz is None:
        return ts.tz_localize(tz)
    return ts.tz_convert(tz)


def build_alert_log(
    *,
    latest_view: pd.DataFrame,
    revision_detail: pd.DataFrame | None = None,
    now: pd.Timestamp | None = None,
    config: ModelConfig | None = None,
) -> pd.DataFrame:
    """Build SLA-based operational alerts from latest-view snapshots."""

    model_config = config or load_config()
    ops_config = model_config.ops
    operational_now = normalize_operational_now(now=now, ops_config=ops_config)
    required_week_end = required_operational_week_end(
        now=operational_now, ops_config=ops_config
    ).strftime("%Y-%m-%d")
    required_snapshot = _select_week(latest_view, required_week_end)
    latest_available = (
        latest_view.sort_values("week_end_ts", kind="mergesort").iloc[-1]
        if not latest_view.empty
        else None
    )
    alerts: list[dict[str, Any]] = []
    generated_at = operational_now.tz_convert("UTC").isoformat().replace("+00:00", "Z")

    if required_snapshot is None:
        latest_week = latest_available["week_end"] if latest_available is not None else None
        alerts.append(
            _alert_row(
                generated_at=generated_at,
                operational_now=operational_now,
                required_week_end=required_week_end,
                snapshot_week_end=latest_week,
                category="data_health",
                level="block",
                code="missing_required_snapshot",
                message=(
                    f"required week_end {required_week_end} is not published; "
                    f"latest available week_end is {latest_week}"
                ),
            )
        )
        return _finalize_alerts(alerts)

    payload = required_snapshot["payload"]
    operational_asof = compute_operational_asof(required_week_end, ops_config)
    published_at = _as_timestamp(required_snapshot["published_at"])
    if published_at > operational_asof:
        alerts.append(
            _alert_row(
                generated_at=generated_at,
                operational_now=operational_now,
                required_week_end=required_week_end,
                snapshot_week_end=required_week_end,
                category="data_health",
                level="warn",
                code="late_snapshot_publication",
                message=(
                    f"snapshot for week_end {required_week_end} published at "
                    f"{_ts_text(published_at)} after SLA cutoff {_ts_text(operational_asof)}"
                ),
            )
        )

    if not _has_valid_signal(payload):
        alerts.append(
            _alert_row(
                generated_at=generated_at,
                operational_now=operational_now,
                required_week_end=required_week_end,
                snapshot_week_end=required_week_end,
                category="signal_validity",
                level="block",
                code="signal_invalid",
                message="signal tuple is incomplete for the required week_end",
            )
        )

    execution_state = str(payload.get("execution_state") or "")
    if execution_state == "degrade":
        alerts.append(
            _alert_row(
                generated_at=generated_at,
                operational_now=operational_now,
                required_week_end=required_week_end,
                snapshot_week_end=required_week_end,
                category="execution_readiness",
                level="degrade",
                code="execution_degraded",
                message=str(
                    payload.get("degraded_reason")
                    or payload.get("execution_block_reason")
                    or "execution state is degrade"
                ),
            )
        )
    elif execution_state == "block":
        alerts.append(
            _alert_row(
                generated_at=generated_at,
                operational_now=operational_now,
                required_week_end=required_week_end,
                snapshot_week_end=required_week_end,
                category="execution_readiness",
                level="block",
                code="execution_blocked",
                message=str(
                    payload.get("execution_block_reason")
                    or payload.get("degraded_reason")
                    or "execution state is block"
                ),
            )
        )

    for record in payload.get("freshness", []):
        if bool(record.get("fresh_enough")):
            continue
        level = str(record.get("blocking_level") or "warn")
        if level not in {"warn", "degrade", "block"}:
            level = "warn"
        code = f"stale_source_{level}"
        alerts.append(
            _alert_row(
                generated_at=generated_at,
                operational_now=operational_now,
                required_week_end=required_week_end,
                snapshot_week_end=required_week_end,
                category="data_health",
                level=level,
                code=code,
                message=(
                    f"{record.get('source_label')} stale: "
                    f"{record.get('reason') or 'freshness gate failed'}"
                ),
            )
        )

    revision_row = _select_revision_week(revision_detail, required_week_end)
    if revision_row is not None and bool(revision_row["material_revision"]):
        alerts.append(
            _alert_row(
                generated_at=generated_at,
                operational_now=operational_now,
                required_week_end=required_week_end,
                snapshot_week_end=required_week_end,
                category="signal_validity",
                level="warn",
                code="material_revision_latest_week",
                message=str(
                    revision_row["revision_reason"]
                    or "latest required week has a material revision"
                ),
            )
        )

    return _finalize_alerts(alerts)


def write_alert_log(
    alert_log: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> AlertArtifacts:
    """Write alert log CSV."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    alert_log_path = out / "alert_log.csv"
    alert_log.to_csv(alert_log_path, index=False)
    return AlertArtifacts(alert_log_path=alert_log_path)


def _select_week(frame: pd.DataFrame, week_end: str) -> pd.Series | None:
    if frame.empty:
        return None
    matches = frame.loc[frame["week_end"] == week_end]
    if matches.empty:
        return None
    return matches.iloc[0]


def _select_revision_week(frame: pd.DataFrame | None, week_end: str) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    matches = frame.loc[frame["week_end"] == week_end]
    if matches.empty:
        return None
    return matches.iloc[0]


def _has_valid_signal(payload: dict[str, Any]) -> bool:
    required_fields = ("k_hat_t", "p_t", "s_t")
    return all(payload.get(field) is not None for field in required_fields)


def _alert_row(
    *,
    generated_at: str,
    operational_now: pd.Timestamp,
    required_week_end: str,
    snapshot_week_end: str | None,
    category: str,
    level: str,
    code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "operational_now": _ts_text(operational_now),
        "required_week_end": required_week_end,
        "snapshot_week_end": snapshot_week_end,
        "category": category,
        "alert_level": level,
        "alert_code": code,
        "message": message,
        "runbook_section": _RUNBOOK_SECTION_BY_CODE.get(code, "§1.1"),
    }


def _finalize_alerts(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "generated_at",
                "operational_now",
                "required_week_end",
                "snapshot_week_end",
                "category",
                "alert_level",
                "alert_code",
                "message",
                "runbook_section",
            ]
        )
    frame = pd.DataFrame(rows)
    frame["_severity_rank"] = frame["alert_level"].map(SEVERITY_ORDER)
    frame = frame.sort_values(
        ["_severity_rank", "category", "alert_code"],
        ascending=[False, True, True],
        kind="mergesort",
    ).drop(columns="_severity_rank")
    return frame.reset_index(drop=True)


def _ts_text(value: pd.Timestamp) -> str:
    ts = _as_timestamp(value)
    if ts.tz is None:
        raise ValueError("timestamp must be timezone-aware")
    return ts.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _as_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tz is None:
        raise ValueError("timestamp must be timezone-aware")
    return ts
