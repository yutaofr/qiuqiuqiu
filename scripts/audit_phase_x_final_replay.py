#!/usr/bin/env python3
"""Audit Phase X final replay artifacts for controlled backfill closure."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qqq_cycle.data_contracts.backfill_validation import validate_normalized_holdings


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
    pre_hold_state_manifest: Path | None
    normalized_path: Path
    price_namespace_path: Path
    proof_path: Path
    strict_upgrade_evaluation_path: Path | None


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


def _live_log_has_controlled_resolution(log_path: Path, week_end: str) -> bool:
    if not log_path.exists():
        return False
    with log_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("week_end")) != week_end:
                continue
            if str(row.get("contract_source")) in {"controlled_block", "stores_backfill", "stores_strict"}:
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
    pre_hold = (
        _load_json(inputs.pre_hold_state_manifest)
        if inputs.pre_hold_state_manifest and inputs.pre_hold_state_manifest.exists()
        else None
    )
    state_continuity_ok = False
    h_t_lead_prev_unchanged = False
    heal_count_unchanged = False
    if pre_hold and state_hold:
        h_t_lead_prev_unchanged = (
            float(state_hold["h_t_lead_prev"]) == float(pre_hold["h_t_lead_prev"])
        )
        heal_count_unchanged = int(state_hold["heal_count"]) == int(pre_hold["heal_count"])
    elif state_hold and state_next:
        h_t_lead_prev_unchanged = (
            float(state_next["h_t_lead_prev"]) == float(state_hold["h_t_lead_prev"])
        )
        heal_count_unchanged = int(state_next["heal_count"]) == int(state_hold["heal_count"])
        state_continuity_ok = h_t_lead_prev_unchanged and heal_count_unchanged
    if pre_hold and state_hold:
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
        contract_source = str(state_hold.get("contract_source"))
        cache_bypass_closed = (
            contract_source in {"controlled_block", "stores_backfill", "stores_strict"}
            and contract_source != "legacy_cache_micro"
        )
    if not cache_bypass_closed:
        cache_bypass_closed = _live_log_has_controlled_resolution(
            inputs.outputs_live_dir / "live_run_log.csv", inputs.week_end
        )

    required_ok = all(artifact_flags.values())
    history_contains_week = _history_contains_week(history_dir, inputs.week_end)
    history_contains_next_week = _history_contains_week(history_dir, inputs.next_week_end)
    no_store_pollution = not store_flags["strict_namespace_polluted"]

    proof = _load_json(inputs.proof_path)
    strict_upgrade = (
        _load_json(inputs.strict_upgrade_evaluation_path)
        if inputs.strict_upgrade_evaluation_path and inputs.strict_upgrade_evaluation_path.exists()
        else {}
    )
    normalized = pd.read_csv(inputs.normalized_path)
    price_namespace = pd.read_csv(inputs.price_namespace_path)
    validation = validate_normalized_holdings(normalized, price_namespace)
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
        "selected_scheme": scheme,
        "proof_strict_eligible": bool(controlled.get("strict_eligible", False)),
        "strict_eligibility_reason": proof.get("strict_eligibility_reason"),
        "weight_sum": validation.weight_sum,
        "unresolved_weight_sum": validation.unresolved_weight_sum,
        "join_coverage_weight": validation.join_coverage_weight,
        "strict_validation_ok": validation.strict_validation_ok,
        "degraded_validation_ok": validation.degraded_validation_ok,
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
        "h_t_lead_prev_unchanged_during_2026_04_24": h_t_lead_prev_unchanged,
        "heal_count_unchanged_during_2026_04_24": heal_count_unchanged,
        "2026_05_01_state_continuity_ok": state_continuity_ok,
        "phase14_history_contains_2026_04_24": history_contains_week,
        "phase14_history_contains_2026_05_01": history_contains_next_week,
        "strict_namespace_polluted": store_flags["strict_namespace_polluted"],
        "cache_micro_bypass_closed": cache_bypass_closed,
        "revision_audit_passed": revision_passed,
        "ops_status_written": bool(
            artifact_flags["ops_status_json_exists"] and artifact_flags["ops_status_md_exists"]
        ),
        "strict_upgrade_attempted": bool(strict_upgrade.get("strict_upgrade_attempted", False)),
        "strict_upgrade_succeeded": bool(strict_upgrade.get("strict_upgrade_succeeded", False)),
        "candidate_strict_eligible": strict_upgrade.get("candidate_strict_eligible"),
        "strict_evidence_class": strict_upgrade.get("evidence_class"),
        "strict_evidence_timestamp_utc": strict_upgrade.get("evidence_timestamp_utc"),
        "raw_content_sha256_match": strict_upgrade.get("raw_content_sha256_match"),
        "extracted_payload_hash_match": strict_upgrade.get("extracted_payload_hash_match"),
        "canonical_hash_match": strict_upgrade.get("canonical_hash_match"),
        "payload_extraction_method": strict_upgrade.get("payload_extraction_method"),
        "canonicalization_method": strict_upgrade.get("canonicalization_method"),
        "canonicalization_version": strict_upgrade.get("canonicalization_version"),
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
    parser.add_argument(
        "--pre-hold-state-manifest",
        type=Path,
        default=None,
        help="Optional pre-hold manifest snapshot for hold-week continuity checks",
    )
    parser.add_argument(
        "--normalized-path",
        type=Path,
        default=Path("normalized/qqq_holdings_2026-04-24_normalized.csv"),
    )
    parser.add_argument(
        "--price-namespace-path",
        type=Path,
        default=Path("data/price_namespace/qqq_price_namespace.csv"),
    )
    parser.add_argument(
        "--proof-path",
        type=Path,
        default=Path("captures/invesco_qqq_holdings_2026-04-24_proof.json"),
    )
    parser.add_argument(
        "--strict-upgrade-evaluation-path",
        type=Path,
        default=Path("outputs/phase14/strict_upgrade_evaluation_2026-04-24.json"),
    )
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
        pre_hold_state_manifest=args.pre_hold_state_manifest,
        normalized_path=args.normalized_path,
        price_namespace_path=args.price_namespace_path,
        proof_path=args.proof_path,
        strict_upgrade_evaluation_path=args.strict_upgrade_evaluation_path,
    )
    report = run_audit(inputs)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["final_audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
