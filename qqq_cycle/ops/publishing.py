"""Phase 14 immutable publication helpers.

This module turns the latest live run summary into immutable historical
Phase 14 artifacts. It does not recompute any signal math. All snapshot fields
are derived from already-materialized live outputs produced at the weekly
decision timestamp.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_OPERATIONAL_SLA_CUTOFF = "SAT 12:00 America/New_York"

_LATEST_SNAPSHOT_NAME = "cycle_snapshot_latest.json"
_LATEST_REPORT_NAME = "weekly_cycle_report_latest.md"

_REQUIRED_SUMMARY_FIELDS = {
    "week_end",
    "mode",
    "k_hat_t",
    "p_t",
    "s_t",
    "h_t",
    "rho_t",
    "I_t",
    "execution_state",
    "execution_permitted",
    "signal_valid_but_not_executable",
    "execution_block_reason",
    "strict_contracts_satisfied",
    "freshness",
    "interpretability",
}


class PublishingInputError(RuntimeError):
    """Raised when required publish inputs are missing or malformed."""


@dataclass(frozen=True)
class CycleSnapshot:
    """Immutable Phase 14 cycle snapshot for one published weekly decision.

    Input: already-materialized live run summary fields from the current week.
    Output: JSON-serializable immutable publication record.
    Time/as-of semantics: all signal fields must have been knowable by the
    decision week_end; this publisher only adds publication metadata.
    """

    week_end: str
    published_at: str
    operational_sla_cutoff: str
    source_hash: str
    mode: str
    k_hat_t: int | None
    p_t: list[float] | None
    s_t: float | None
    h_t: float | None
    rho_t: float | None
    drift_flag: int | None
    I_t: dict[str, Any] | None
    degraded_reason: str | None
    execution_state: str | None
    execution_permitted: bool | None
    signal_valid_but_not_executable: bool | None
    execution_block_reason: str | None
    strict_contracts_satisfied: bool | None
    freshness: list[dict[str, Any]]
    interpretability: dict[str, Any] | None


@dataclass(frozen=True)
class Phase14PublishArtifacts:
    """Paths written by one immutable Phase 14 publication run."""

    snapshot_history_path: Path
    snapshot_latest_path: Path
    report_history_path: Path
    report_latest_path: Path


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=True)


def _normalize_published_at(published_at: str | None = None) -> str:
    if published_at is None:
        dt = datetime.now(timezone.utc)
    else:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("published_at must include timezone information")
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _filename_timestamp(published_at: str) -> str:
    return published_at.replace(":", "-")


def load_live_run_summary(summary_path: str | Path) -> dict[str, Any]:
    """Load and validate the live run summary used as the publish source."""

    path = Path(summary_path)
    if not path.exists():
        raise PublishingInputError(
            f"live run summary not found at {path}. "
            "Run scripts/run_live_pipeline.py before Phase 14 publishing."
        )
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PublishingInputError(f"live run summary is not valid JSON: {path}") from exc

    missing = sorted(_REQUIRED_SUMMARY_FIELDS.difference(summary))
    if missing:
        joined = ", ".join(missing)
        raise PublishingInputError(
            "live run summary is missing required Phase 14 publish fields: "
            f"{joined}. Re-run scripts/run_live_pipeline.py with the updated artifact schema."
        )
    return summary


def build_cycle_snapshot(
    live_summary: dict[str, Any],
    *,
    published_at: str | None = None,
    operational_sla_cutoff: str = DEFAULT_OPERATIONAL_SLA_CUTOFF,
) -> CycleSnapshot:
    """Wrap one live summary in immutable publication metadata."""

    normalized_published_at = _normalize_published_at(published_at)
    source_hash = hashlib.sha256(_canonical_json(live_summary).encode("utf-8")).hexdigest()

    interpretability = live_summary.get("interpretability") or {}
    drift_flag_raw = interpretability.get("drift_flag")
    drift_flag = int(drift_flag_raw) if drift_flag_raw is not None else None

    p_t_raw = live_summary.get("p_t")
    p_t = [float(value) for value in p_t_raw] if isinstance(p_t_raw, list) else None

    return CycleSnapshot(
        week_end=str(live_summary["week_end"]),
        published_at=normalized_published_at,
        operational_sla_cutoff=operational_sla_cutoff,
        source_hash=source_hash,
        mode=str(live_summary["mode"]),
        k_hat_t=int(live_summary["k_hat_t"]) if live_summary["k_hat_t"] is not None else None,
        p_t=p_t,
        s_t=_maybe_float(live_summary.get("s_t")),
        h_t=_maybe_float(live_summary.get("h_t")),
        rho_t=_maybe_float(live_summary.get("rho_t")),
        drift_flag=drift_flag,
        I_t=live_summary.get("I_t"),
        degraded_reason=_maybe_str(live_summary.get("degraded_reason")),
        execution_state=_maybe_str(live_summary.get("execution_state")),
        execution_permitted=_maybe_bool(live_summary.get("execution_permitted")),
        signal_valid_but_not_executable=_maybe_bool(
            live_summary.get("signal_valid_but_not_executable")
        ),
        execution_block_reason=_maybe_str(live_summary.get("execution_block_reason")),
        strict_contracts_satisfied=_maybe_bool(live_summary.get("strict_contracts_satisfied")),
        freshness=_normalize_freshness(live_summary.get("freshness")),
        interpretability=interpretability or None,
    )


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _maybe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _normalize_freshness(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise PublishingInputError("freshness must be a list of source status records")
    return [dict(record) for record in value]


def render_weekly_cycle_report(snapshot: CycleSnapshot) -> str:
    """Render the human-readable weekly report from the immutable snapshot."""

    lines = [
        "# Weekly Cycle Report",
        "",
        "## Snapshot",
        "",
        f"- week_end: {snapshot.week_end}",
        f"- published_at: {snapshot.published_at}",
        f"- operational_sla_cutoff: {snapshot.operational_sla_cutoff}",
        f"- source_hash: {snapshot.source_hash}",
        f"- mode: {snapshot.mode}",
        f"- k_hat_t: {_scalar_to_text(snapshot.k_hat_t)}",
        f"- p_t: {json.dumps(snapshot.p_t, ensure_ascii=True)}",
        f"- s_t: {_scalar_to_text(snapshot.s_t)}",
        f"- h_t: {_scalar_to_text(snapshot.h_t)}",
        f"- rho_t: {_scalar_to_text(snapshot.rho_t)}",
        f"- drift_flag: {_scalar_to_text(snapshot.drift_flag)}",
        f"- execution_state: {_scalar_to_text(snapshot.execution_state)}",
        f"- execution_permitted: {_scalar_to_text(snapshot.execution_permitted)}",
        (
            "- signal_valid_but_not_executable: "
            f"{_scalar_to_text(snapshot.signal_valid_but_not_executable)}"
        ),
        (
            "- strict_contracts_satisfied: "
            f"{_scalar_to_text(snapshot.strict_contracts_satisfied)}"
        ),
        f"- degraded_reason: {_scalar_to_text(snapshot.degraded_reason)}",
        f"- execution_block_reason: {_scalar_to_text(snapshot.execution_block_reason)}",
        "",
        "## Freshness",
        "",
    ]

    if snapshot.freshness:
        for record in snapshot.freshness:
            lines.append(
                "- "
                f"{record.get('source_label', 'unknown')}: "
                f"fresh_enough={record.get('fresh_enough')}, "
                f"blocking_level={record.get('blocking_level')}, "
                f"last_observation_date={record.get('last_observation_date')}, "
                f"reason={record.get('reason')}"
            )
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def publish_cycle_snapshot(
    snapshot: CycleSnapshot,
    output_dir: str | Path,
) -> Phase14PublishArtifacts:
    """Write immutable history artifacts and overwrite only the latest pointers."""

    phase14_dir = Path(output_dir)
    history_dir = phase14_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    timestamp_token = _filename_timestamp(snapshot.published_at)
    snapshot_history_path = history_dir / (
        f"cycle_snapshot_{snapshot.week_end}__run_{timestamp_token}.json"
    )
    report_history_path = phase14_dir / (
        f"weekly_cycle_report_{snapshot.week_end}__run_{timestamp_token}.md"
    )

    if snapshot_history_path.exists():
        raise FileExistsError(
            f"immutable history snapshot already exists: {snapshot_history_path}"
        )
    if report_history_path.exists():
        raise FileExistsError(
            f"immutable history report already exists: {report_history_path}"
        )

    snapshot_latest_path = phase14_dir / _LATEST_SNAPSHOT_NAME
    report_latest_path = phase14_dir / _LATEST_REPORT_NAME

    snapshot_json = _canonical_json(asdict(snapshot)) + "\n"
    report_markdown = render_weekly_cycle_report(snapshot)

    snapshot_history_path.write_text(snapshot_json, encoding="utf-8")
    report_history_path.write_text(report_markdown, encoding="utf-8")
    snapshot_latest_path.write_text(snapshot_json, encoding="utf-8")
    report_latest_path.write_text(report_markdown, encoding="utf-8")

    return Phase14PublishArtifacts(
        snapshot_history_path=snapshot_history_path,
        snapshot_latest_path=snapshot_latest_path,
        report_history_path=report_history_path,
        report_latest_path=report_latest_path,
    )


def publish_from_live_summary_path(
    *,
    summary_path: str | Path,
    output_dir: str | Path,
    published_at: str | None = None,
    operational_sla_cutoff: str = DEFAULT_OPERATIONAL_SLA_CUTOFF,
) -> tuple[CycleSnapshot, Phase14PublishArtifacts]:
    """Load one live summary and publish immutable Phase 14 artifacts."""

    live_summary = load_live_run_summary(summary_path)
    snapshot = build_cycle_snapshot(
        live_summary,
        published_at=published_at,
        operational_sla_cutoff=operational_sla_cutoff,
    )
    artifacts = publish_cycle_snapshot(snapshot, output_dir)
    return snapshot, artifacts


def _scalar_to_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)
