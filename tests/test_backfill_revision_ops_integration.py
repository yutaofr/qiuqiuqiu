from __future__ import annotations

import json

import pandas as pd

from qqq_cycle.data_contracts.backfill_validation import BackfillValidationResult
from qqq_cycle.data_contracts.publication_proof import PublicationProof
from qqq_cycle.ops.backfill_ingest import (
    BackfillDecision,
    controlled_backfill_result_from_decision,
)
from qqq_cycle.ops.revision_audit import append_controlled_backfill_revision_record
from qqq_cycle.ops.status import build_ops_status_summary


def proof(strict: bool) -> PublicationProof:
    return PublicationProof(
        source_label="Invesco official QQQ holdings export",
        source_url="https://www.invesco.com/us/financial-products/etfs/holdings?ticker=QQQ",
        content_sha256="a" * 64,
        fetched_at_utc="2026-04-25T15:00:00Z",
        evidence_class="direct_http_capture_at_or_before_sla",
        evidence_timestamp_utc="2026-04-25T15:00:00Z",
        http_status=200,
        http_date_header=None,
        etag=None,
        last_modified_header=None,
        object_version_id=None,
        audit_log_sha256=None,
        third_party_snapshot_url=None,
        strict_eligible=strict,
        strict_eligibility_reason="verified_direct_http_capture_before_sla" if strict else "evidence_after_sla_cutoff",
    )


def validation(reason: str, strict: bool = False, degraded: bool = False) -> BackfillValidationResult:
    return BackfillValidationResult(
        weight_sum=1.0,
        weight_sum_ok=strict,
        join_coverage_weight=1.0,
        join_coverage_ok=strict,
        unresolved_weight_sum=0.0,
        unresolved_weight_ok=degraded,
        strict_validation_ok=strict,
        degraded_validation_ok=degraded,
        validation_reason=reason,
    )


def normalized() -> pd.DataFrame:
    return pd.DataFrame([{"instrument_id": "I1", "normalized_weight": 1.0}])


def result_for(decision: BackfillDecision, validation_reason: str) -> dict:
    result = controlled_backfill_result_from_decision(
        week_end="2026-04-24",
        asset="QQQ",
        proof_result=proof(decision.strict_eligible),
        validation_result=validation(
            validation_reason,
            strict=decision.strict_validation_ok,
            degraded=decision.degraded_validation_ok,
        ),
        decision=decision,
        normalized_holdings=normalized(),
        created_at_utc="2026-04-29T00:00:00Z",
    )
    return result.to_dict()


def test_revision_audit_records_strict_recovery_with_verified_pit_reason(tmp_path) -> None:
    payload = result_for(
        BackfillDecision("strict_recovery", "strict_recovery_verified_pit_availability", True, True, True),
        "validation_passed_strict",
    )

    path = append_controlled_backfill_revision_record(payload, output_dir=tmp_path)
    record = json.loads(path.read_text().strip())

    assert record["backfill_mode"] == "strict_recovery"
    assert record["revision_reason"] == "controlled_backfill_with_verified_pit_availability"


def test_revision_audit_records_degraded_backfill_without_pit_proof_reason(tmp_path) -> None:
    payload = result_for(
        BackfillDecision("degraded_backfill", "degraded_backfill_without_pit_proof", False, True, True),
        "validation_passed_strict",
    )

    path = append_controlled_backfill_revision_record(payload, output_dir=tmp_path)
    record = json.loads(path.read_text().strip())

    assert record["backfill_mode"] == "degraded_backfill"
    assert record["revision_reason"] == "controlled_backfill_without_pit_proof"


def test_revision_audit_records_block_from_namespace_failure(tmp_path) -> None:
    payload = result_for(
        BackfillDecision("block", "block_normalization_failure", False, False, False),
        "normalization_failure",
    )

    path = append_controlled_backfill_revision_record(payload, output_dir=tmp_path)
    record = json.loads(path.read_text().strip())

    assert record["backfill_mode"] == "block"
    assert record["revision_reason"] == "namespace_normalization_failure"


def test_revision_audit_records_block_from_weight_sum_violation(tmp_path) -> None:
    payload = result_for(
        BackfillDecision("block", "block_weight_sum_violation", False, False, False),
        "weight_sum_violation",
    )

    path = append_controlled_backfill_revision_record(payload, output_dir=tmp_path)
    record = json.loads(path.read_text().strip())

    assert record["revision_reason"] == "weight_sum_violation"


def test_revision_audit_records_block_from_insufficient_join_coverage(tmp_path) -> None:
    payload = result_for(
        BackfillDecision("block", "block_insufficient_join_coverage", False, False, False),
        "insufficient_join_coverage",
    )

    path = append_controlled_backfill_revision_record(payload, output_dir=tmp_path)
    record = json.loads(path.read_text().strip())

    assert record["revision_reason"] == "insufficient_join_coverage"


def test_revision_audit_does_not_alter_decision() -> None:
    payload = result_for(
        BackfillDecision("degraded_backfill", "degraded_backfill_without_pit_proof", False, True, True),
        "validation_passed_strict",
    )

    assert payload["backfill_mode"] == "degraded_backfill"


def test_ops_status_summary_exposes_backfill_mode_and_strict_eligible() -> None:
    payload = result_for(
        BackfillDecision("degraded_backfill", "degraded_backfill_without_pit_proof", False, True, True),
        "validation_passed_strict",
    )
    alert_log = pd.DataFrame(columns=["alert_level", "category", "alert_code", "message", "runbook_section"])

    summary = build_ops_status_summary(
        latest_view=pd.DataFrame(),
        alert_log=alert_log,
        controlled_backfill_result=payload,
        now=pd.Timestamp("2026-04-29T12:00:00Z"),
    )

    assert summary["controlled_backfill"]["backfill_mode"] == "degraded_backfill"
    assert summary["controlled_backfill"]["strict_eligible"] is False
