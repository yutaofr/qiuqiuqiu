#!/usr/bin/env python3
"""Audit Phase X final replay artifacts for controlled backfill closure."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditInputs:
    week_end: str
    next_week_end: str
    asset: str
    captures_dir: Path
    outputs_phase14_dir: Path
    outputs_live_dir: Path
    state_dir: Path
    store_root: Path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _history_latest_for_week(history_dir: Path, week_end: str) -> dict[str, Any] | None:
    matches = sorted(history_dir.glob(f"cycle_snapshot_{week_end}__run_*.json"))
    if not matches:
        return None
    return _load_json(matches[-1])


def _state_manifest(state_dir: Path, week_end: str) -> dict[str, Any] | None:
    dated = state_dir / f"live_state_{week_end.replace('-', '')}" / "manifest.json"
    if dated.exists():
        return _load_json(dated)
    return None


def _read_controlled_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"controlled result missing: {path}")
    return _load_json(path)


def _required_artifacts_exist(inputs: AuditInputs) -> dict[str, bool]:
    week = inputs.week_end
    cap = inputs.captures_dir
    p14 = inputs.outputs_phase14_dir
    live = inputs.outputs_live_dir
    raw_candidates = sorted(cap.glob(f"invesco_qqq_holdings_{week}_raw.*"))
    return {
        "raw_capture_exists": bool(raw_candidates),
        "proof_exists": (cap / f"invesco_qqq_holdings_{week}_proof.json").exists(),
        "capture_status_exists": (cap / f"invesco_qqq_holdings_{week}_capture_status.json").exists(),
        "controlled_result_exists": (
            p14 / f"controlled_backfill_result_{inputs.asset.lower()}_{week}.json"
        ).exists(),
        "controlled_revisions_exists": (
            p14 / f"controlled_backfill_revisions_{inputs.asset.lower()}_{week}.jsonl"
        ).exists(),
        "live_run_summary_exists": (live / "live_run_summary.json").exists(),
        "live_run_log_exists": (live / "live_run_log.csv").exists(),
        "phase14_snapshot_latest_exists": (p14 / "cycle_snapshot_latest.json").exists(),
        "phase14_history_exists": (p14 / "history").exists(),
        "revision_summary_exists": (p14 / "revision_stability_summary.csv").exists(),
        "revision_detail_exists": (p14 / "revision_stability_detail.csv").exists(),
        "revision_tests_exists": (p14 / "revision_stability_tests.json").exists(),
        "ops_status_json_exists": (p14 / "ops_status_summary.json").exists(),
        "ops_status_md_exists": (p14 / "ops_status_summary.md").exists(),
    }


def _history_contains_week(history_dir: Path, week_end: str) -> bool:
    return any(history_dir.glob(f"cycle_snapshot_{week_end}__run_*.json"))


def _store_flags(inputs: AuditInputs, scheme: str) -> dict[str, bool]:
    week = inputs.week_end
    asset_lc = inputs.asset.lower()
    strict_const = inputs.store_root / "strict" / "constituents" / f"{asset_lc}_constituents_{week}.csv"
    strict_weights = inputs.store_root / "strict" / "weights" / f"{asset_lc}_weights_{week}.csv"
    backfill_const = inputs.store_root / "backfill" / "constituents" / f"{asset_lc}_constituents_{week}.csv"
    backfill_weights = inputs.store_root / "backfill" / "weights" / f"{asset_lc}_weights_{week}.csv"
    strict_exists = strict_const.exists() or strict_weights.exists()
    backfill_exists = backfill_const.exists() or backfill_weights.exists()
    strict_namespace_polluted = False
    if scheme == "degraded_backfill":
        strict_namespace_polluted = strict_exists
    if scheme == "block":
        strict_namespace_polluted = strict_exists or backfill_exists
    return {
        "strict_store_exists_for_week": strict_exists,
        "backfill_store_exists_for_week": backfill_exists,
        "strict_namespace_polluted": strict_namespace_polluted,
    }


def _revision_audit_passed(path: Path) -> bool:
    if not path.exists():
        return False
    payload = _load_json(path)
    checks = payload.get("checks", {})
    return bool(checks) and all(bool(value) for value in checks.values())


def _live_log_has_controlled_block(log_path: Path, week_end: str) -> bool:
    if not log_path.exists():
        return False
    with log_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("week_end")) != week_end:
                continue
            if str(row.get("contract_source")) == "controlled_block":
                return True
    return False


def run_audit(inputs: AuditInputs) -> dict[str, Any]:
    history_dir = inputs.outputs_phase14_dir / "history"
    controlled_path = (
        inputs.outputs_phase14_dir
        / f"controlled_backfill_result_{inputs.asset.lower()}_{inputs.week_end}.json"
    )
    controlled = _read_controlled_result(controlled_path)
    scheme = str(controlled["backfill_mode"])

    state_hold = _state_manifest(inputs.state_dir, inputs.week_end)
    state_next = _state_manifest(inputs.state_dir, inputs.next_week_end)
    state_continuity_ok = False
    h_t_lead_prev_unchanged = False
    heal_count_unchanged = False
    if state_hold and state_next:
        h_t_lead_prev_unchanged = (
            float(state_next["h_t_lead_prev"]) == float(state_hold["h_t_lead_prev"])
        )
        heal_count_unchanged = int(state_next["heal_count"]) == int(state_hold["heal_count"])
        state_continuity_ok = h_t_lead_prev_unchanged and heal_count_unchanged

    week_snapshot = _history_latest_for_week(history_dir, inputs.week_end)
    next_snapshot = _history_latest_for_week(history_dir, inputs.next_week_end)
    artifact_flags = _required_artifacts_exist(inputs)
    store_flags = _store_flags(inputs, scheme)
    revision_passed = _revision_audit_passed(
        inputs.outputs_phase14_dir / "revision_stability_tests.json"
    )
    cache_bypass_closed = False
    if state_hold is not None:
        cache_bypass_closed = (
            str(state_hold.get("contract_source")) == "controlled_block"
            and bool(state_hold.get("strict_gate_passed")) is False
        )
    if not cache_bypass_closed:
        cache_bypass_closed = _live_log_has_controlled_block(
            inputs.outputs_live_dir / "live_run_log.csv", inputs.week_end
        )

    required_ok = all(artifact_flags.values())
    history_contains_week = _history_contains_week(history_dir, inputs.week_end)
    history_contains_next_week = _history_contains_week(history_dir, inputs.next_week_end)
    no_store_pollution = not store_flags["strict_namespace_polluted"]
    final_pass = (
        required_ok
        and history_contains_week
        and history_contains_next_week
        and cache_bypass_closed
        and no_store_pollution
        and revision_passed
        and state_continuity_ok
    )

    return {
        "scheme": scheme,
        "strict_eligible": bool(controlled.get("strict_eligible", False)),
        "validation_reason": controlled.get("validation_reason"),
        "decision_reason": controlled.get("decision_reason"),
        "contract_source_2026_04_24": (
            state_hold.get("contract_source") if state_hold else None
        ),
        "strict_gate_passed_2026_04_24": (
            bool(state_hold.get("strict_gate_passed")) if state_hold else False
        ),
        "micro_state_frozen_2026_04_24": (
            bool(state_hold.get("micro_state_frozen")) if state_hold else False
        ),
        "h_t_2026_04_24": week_snapshot.get("h_t") if week_snapshot else None,
        "rho_t_2026_04_24": week_snapshot.get("rho_t") if week_snapshot else None,
        "h_t_lead_prev_unchanged_on_hold_week": h_t_lead_prev_unchanged,
        "heal_count_unchanged_on_hold_week": heal_count_unchanged,
        "2026_05_01_state_continuity_ok": state_continuity_ok,
        "history_contains_2026_04_24": history_contains_week,
        "history_contains_2026_05_01": history_contains_next_week,
        "strict_namespace_polluted": store_flags["strict_namespace_polluted"],
        "cache_micro_bypass_closed": cache_bypass_closed,
        "revision_audit_passed": revision_passed,
        "ops_status_written": bool(
            artifact_flags["ops_status_json_exists"] and artifact_flags["ops_status_md_exists"]
        ),
        "required_artifacts_ok": required_ok,
        "artifact_flags": artifact_flags,
        "final_audit_passed": final_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week-end", default="2026-04-24")
    parser.add_argument("--next-week-end", default="2026-05-01")
    parser.add_argument("--asset", default="QQQ")
    parser.add_argument("--captures-dir", type=Path, default=Path("captures"))
    parser.add_argument("--outputs-phase14-dir", type=Path, default=Path("outputs/phase14"))
    parser.add_argument("--outputs-live-dir", type=Path, default=Path("outputs/live"))
    parser.add_argument("--state-dir", type=Path, default=Path("state"))
    parser.add_argument("--store-root", type=Path, default=Path("stores"))
    args = parser.parse_args()

    inputs = AuditInputs(
        week_end=args.week_end,
        next_week_end=args.next_week_end,
        asset=args.asset,
        captures_dir=args.captures_dir,
        outputs_phase14_dir=args.outputs_phase14_dir,
        outputs_live_dir=args.outputs_live_dir,
        state_dir=args.state_dir,
        store_root=args.store_root,
    )
    report = run_audit(inputs)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["final_audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
