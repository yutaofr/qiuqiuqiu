"""Phase 14 immutable-history revision audit.

The audit reads timestamped history snapshots only. It never relies on
overwrite behavior. For each week_end it compares the earliest published
snapshot to the latest published snapshot and emits per-week detail plus
aggregate stability summaries.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

MATERIAL_DELTA_THRESHOLD = 0.05
_FLOAT_TOLERANCE = 1e-12

_REQUIRED_SNAPSHOT_FIELDS = {
    "week_end",
    "published_at",
    "mode",
    "k_hat_t",
    "p_t",
    "s_t",
    "h_t",
    "rho_t",
    "drift_flag",
    "source_hash",
}


class RevisionAuditInputError(RuntimeError):
    """Raised when immutable Phase 14 history is missing or malformed."""


@dataclass(frozen=True)
class RevisionAuditArtifacts:
    """Paths written by one revision-audit run."""

    summary_csv_path: Path
    detail_csv_path: Path
    tests_json_path: Path


def load_snapshot_history(history_dir: str | Path) -> pd.DataFrame:
    """Load immutable snapshot history into a typed DataFrame.

    Input: Phase 14 immutable history directory containing timestamped JSON
    snapshots. Output: one row per snapshot file with published_at parsed in
    UTC and sorted point-in-time publication order.
    """

    path = Path(history_dir)
    if not path.exists():
        raise RevisionAuditInputError(
            f"history directory not found: {path}. Run Phase 14 publishing first."
        )

    rows: list[dict[str, Any]] = []
    for snapshot_path in sorted(path.glob("*.json")):
        try:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RevisionAuditInputError(
                f"history snapshot is not valid JSON: {snapshot_path}"
            ) from exc

        missing = sorted(_REQUIRED_SNAPSHOT_FIELDS.difference(payload))
        if missing:
            joined = ", ".join(missing)
            raise RevisionAuditInputError(
                f"history snapshot missing required fields ({joined}): {snapshot_path}"
            )

        published_at = pd.Timestamp(payload["published_at"])
        if published_at.tz is None:
            raise RevisionAuditInputError(
                f"published_at must be timezone-aware: {snapshot_path}"
            )

        rows.append(
            {
                "snapshot_path": str(snapshot_path),
                "week_end": str(payload["week_end"]),
                "published_at": published_at.tz_convert("UTC"),
                "mode": payload.get("mode"),
                "k_hat_t": payload.get("k_hat_t"),
                "p_t": payload.get("p_t"),
                "s_t": _maybe_float(payload.get("s_t")),
                "h_t": _maybe_float(payload.get("h_t")),
                "rho_t": _maybe_float(payload.get("rho_t")),
                "drift_flag": payload.get("drift_flag"),
                "source_hash": payload.get("source_hash"),
                "payload": payload,
            }
        )

    if not rows:
        raise RevisionAuditInputError(
            f"no history snapshots found in {path}. Run Phase 14 publishing first."
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["week_end", "published_at", "snapshot_path"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_revision_detail(history_frame: pd.DataFrame) -> pd.DataFrame:
    """Compare earliest and latest immutable snapshots for each week_end."""

    detail_rows: list[dict[str, Any]] = []

    for week_end, group in history_frame.groupby("week_end", sort=True):
        ordered = group.sort_values(["published_at", "snapshot_path"], kind="mergesort").reset_index(drop=True)
        first = ordered.iloc[0]
        latest = ordered.iloc[-1]

        delta_s = _delta(first["s_t"], latest["s_t"])
        delta_h = _delta(first["h_t"], latest["h_t"])
        delta_rho = _delta(first["rho_t"], latest["rho_t"])
        delta_p_max = _max_abs_list_delta(first["p_t"], latest["p_t"])

        mode_changed = first["mode"] != latest["mode"]
        k_hat_changed = first["k_hat_t"] != latest["k_hat_t"]
        p_t_changed = not _lists_close(first["p_t"], latest["p_t"])
        drift_flag_changed = first["drift_flag"] != latest["drift_flag"]

        reasons = _revision_reasons(
            mode_changed=mode_changed,
            k_hat_changed=k_hat_changed,
            p_t_changed=p_t_changed,
            drift_flag_changed=drift_flag_changed,
            delta_s=delta_s,
            delta_h=delta_h,
            delta_rho=delta_rho,
        )

        material_revision = _is_material_revision(
            mode_changed=mode_changed,
            k_hat_changed=k_hat_changed,
            delta_s=delta_s,
            delta_h=delta_h,
            delta_rho=delta_rho,
        )

        detail_rows.append(
            {
                "week_end": week_end,
                "first_published_at": _ts_text(first["published_at"]),
                "latest_published_at": _ts_text(latest["published_at"]),
                "run_count": int(len(ordered)),
                "initial_mode": first["mode"],
                "latest_mode": latest["mode"],
                "initial_k_hat": first["k_hat_t"],
                "latest_k_hat": latest["k_hat_t"],
                "delta_s": delta_s,
                "delta_h": delta_h,
                "delta_rho": delta_rho,
                "material_revision": bool(material_revision),
                "revision_reason": "; ".join(reasons),
                "p_t_changed": bool(p_t_changed),
                "delta_p_max": delta_p_max,
                "drift_flag_changed": bool(drift_flag_changed),
                "initial_drift_flag": first["drift_flag"],
                "latest_drift_flag": latest["drift_flag"],
                "initial_source_hash": first["source_hash"],
                "latest_source_hash": latest["source_hash"],
                "initial_snapshot_path": first["snapshot_path"],
                "latest_snapshot_path": latest["snapshot_path"],
            }
        )

    detail = pd.DataFrame(detail_rows)
    return detail.sort_values("week_end", kind="mergesort").reset_index(drop=True)


def build_revision_summary(detail_frame: pd.DataFrame) -> pd.DataFrame:
    """Build aggregate stability summary rows from per-week detail."""

    if detail_frame.empty:
        return pd.DataFrame(
            [
                {
                    "weeks_total": 0,
                    "weeks_with_multiple_runs": 0,
                    "material_revision_weeks": 0,
                    "mode_changed_weeks": 0,
                    "k_hat_changed_weeks": 0,
                    "p_t_changed_weeks": 0,
                    "drift_flag_changed_weeks": 0,
                    "max_abs_delta_s": 0.0,
                    "max_abs_delta_h": 0.0,
                    "max_abs_delta_rho": 0.0,
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "weeks_total": int(len(detail_frame)),
                "weeks_with_multiple_runs": int((detail_frame["run_count"] > 1).sum()),
                "material_revision_weeks": int(detail_frame["material_revision"].sum()),
                "mode_changed_weeks": int((detail_frame["initial_mode"] != detail_frame["latest_mode"]).sum()),
                "k_hat_changed_weeks": int((detail_frame["initial_k_hat"] != detail_frame["latest_k_hat"]).sum()),
                "p_t_changed_weeks": int(detail_frame["p_t_changed"].sum()),
                "drift_flag_changed_weeks": int(detail_frame["drift_flag_changed"].sum()),
                "max_abs_delta_s": _series_max_abs(detail_frame["delta_s"]),
                "max_abs_delta_h": _series_max_abs(detail_frame["delta_h"]),
                "max_abs_delta_rho": _series_max_abs(detail_frame["delta_rho"]),
            }
        ]
    )


def build_revision_tests(detail_frame: pd.DataFrame) -> dict[str, Any]:
    """Build machine-readable revision-audit checks and thresholds."""

    if detail_frame.empty:
        return {
            "thresholds": _threshold_payload(),
            "counts": {
                "weeks_total": 0,
                "weeks_with_multiple_runs": 0,
                "material_revision_weeks": 0,
            },
            "checks": {
                "earliest_latest_selection_works": False,
                "same_week_multiple_runs_supported": False,
                "material_revision_thresholds_applied": False,
            },
        }

    non_decreasing = bool((pd.to_datetime(detail_frame["first_published_at"], utc=True) <= pd.to_datetime(
        detail_frame["latest_published_at"], utc=True
    )).all())
    has_multi_run_week = bool((detail_frame["run_count"] > 1).any())
    threshold_consistent = bool(
        (
            detail_frame["material_revision"]
            == detail_frame.apply(
                lambda row: _is_material_revision(
                    mode_changed=bool(row["initial_mode"] != row["latest_mode"]),
                    k_hat_changed=bool(row["initial_k_hat"] != row["latest_k_hat"]),
                    delta_s=_maybe_float(row["delta_s"]),
                    delta_h=_maybe_float(row["delta_h"]),
                    delta_rho=_maybe_float(row["delta_rho"]),
                ),
                axis=1,
            )
        ).all()
    )

    return {
        "thresholds": _threshold_payload(),
        "counts": {
            "weeks_total": int(len(detail_frame)),
            "weeks_with_multiple_runs": int((detail_frame["run_count"] > 1).sum()),
            "material_revision_weeks": int(detail_frame["material_revision"].sum()),
        },
        "checks": {
            "earliest_latest_selection_works": non_decreasing,
            "same_week_multiple_runs_supported": has_multi_run_week,
            "material_revision_thresholds_applied": threshold_consistent,
        },
    }


def write_revision_audit_outputs(
    *,
    history_dir: str | Path,
    output_dir: str | Path,
) -> RevisionAuditArtifacts:
    """Run the immutable-history revision audit and write artifacts."""

    history = load_snapshot_history(history_dir)
    detail = build_revision_detail(history)
    summary = build_revision_summary(detail)
    tests_payload = build_revision_tests(detail)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_csv_path = out / "revision_stability_summary.csv"
    detail_csv_path = out / "revision_stability_detail.csv"
    tests_json_path = out / "revision_stability_tests.json"

    summary.to_csv(summary_csv_path, index=False)
    detail.to_csv(detail_csv_path, index=False)
    tests_json_path.write_text(json.dumps(tests_payload, indent=2), encoding="utf-8")

    return RevisionAuditArtifacts(
        summary_csv_path=summary_csv_path,
        detail_csv_path=detail_csv_path,
        tests_json_path=tests_json_path,
    )


def _maybe_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _delta(initial: float | None, latest: float | None) -> float | None:
    if initial is None or latest is None:
        return None
    return float(latest) - float(initial)


def _max_abs_list_delta(initial: Any, latest: Any) -> float | None:
    if not isinstance(initial, list) or not isinstance(latest, list):
        return None
    if len(initial) != len(latest):
        return None
    return max(abs(float(a) - float(b)) for a, b in zip(initial, latest, strict=True))


def _lists_close(initial: Any, latest: Any, *, tol: float = 1e-12) -> bool:
    if initial is None and latest is None:
        return True
    if not isinstance(initial, list) or not isinstance(latest, list):
        return False
    if len(initial) != len(latest):
        return False
    return all(abs(float(a) - float(b)) <= tol for a, b in zip(initial, latest, strict=True))


def _is_material_delta(delta: float | None) -> bool:
    if delta is None:
        return False
    magnitude = abs(delta)
    if math.isclose(magnitude, MATERIAL_DELTA_THRESHOLD, rel_tol=0.0, abs_tol=_FLOAT_TOLERANCE):
        return False
    return magnitude > MATERIAL_DELTA_THRESHOLD


def _is_material_revision(
    *,
    mode_changed: bool,
    k_hat_changed: bool,
    delta_s: float | None,
    delta_h: float | None,
    delta_rho: float | None,
) -> bool:
    return bool(
        mode_changed
        or k_hat_changed
        or _is_material_delta(delta_s)
        or _is_material_delta(delta_h)
        or _is_material_delta(delta_rho)
    )


def _revision_reasons(
    *,
    mode_changed: bool,
    k_hat_changed: bool,
    p_t_changed: bool,
    drift_flag_changed: bool,
    delta_s: float | None,
    delta_h: float | None,
    delta_rho: float | None,
) -> list[str]:
    reasons: list[str] = []
    if mode_changed:
        reasons.append("mode_changed")
    if k_hat_changed:
        reasons.append("k_hat_t_changed")
    if _is_material_delta(delta_s):
        reasons.append("delta_s_gt_0.05")
    if _is_material_delta(delta_h):
        reasons.append("delta_h_gt_0.05")
    if _is_material_delta(delta_rho):
        reasons.append("delta_rho_gt_0.05")
    if p_t_changed:
        reasons.append("p_t_changed")
    if drift_flag_changed:
        reasons.append("drift_flag_changed")
    return reasons


def _ts_text(value: pd.Timestamp) -> str:
    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _series_max_abs(series: pd.Series) -> float:
    finite = [abs(float(value)) for value in series if value is not None and pd.notna(value)]
    if not finite:
        return 0.0
    return float(max(finite))


def _threshold_payload() -> dict[str, Any]:
    return {
        "abs_delta_s_gt": MATERIAL_DELTA_THRESHOLD,
        "abs_delta_h_gt": MATERIAL_DELTA_THRESHOLD,
        "abs_delta_rho_gt": MATERIAL_DELTA_THRESHOLD,
        "mode_changed_is_material": True,
        "k_hat_t_changed_is_material": True,
    }
