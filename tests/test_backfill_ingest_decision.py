from __future__ import annotations

from qqq_cycle.data_contracts.backfill_validation import BackfillValidationResult
from qqq_cycle.data_contracts.publication_proof import PublicationProof
from qqq_cycle.ops.backfill_ingest import decide_backfill_scheme


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
        strict_eligibility_reason=(
            "verified_direct_http_capture_before_sla" if strict else "evidence_after_sla_cutoff"
        ),
    )


def validation(strict: bool, degraded: bool, reason: str = "validation_passed_strict") -> BackfillValidationResult:
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


def test_proof_true_strict_validation_true_selects_strict_recovery() -> None:
    decision = decide_backfill_scheme(proof(True), validation(True, True))

    assert decision.scheme == "strict_recovery"
    assert decision.reason == "strict_recovery_verified_pit_availability"


def test_proof_false_strict_validation_true_selects_degraded_backfill() -> None:
    decision = decide_backfill_scheme(proof(False), validation(True, True))

    assert decision.scheme == "degraded_backfill"
    assert decision.reason == "degraded_backfill_without_pit_proof"


def test_proof_true_strict_false_degraded_true_selects_degraded_backfill() -> None:
    decision = decide_backfill_scheme(
        proof(True), validation(False, True, "validation_passed_degraded_only")
    )

    assert decision.scheme == "degraded_backfill"
    assert decision.reason == "degraded_backfill_validation_only"


def test_proof_false_degraded_true_selects_degraded_backfill() -> None:
    decision = decide_backfill_scheme(
        proof(False), validation(False, True, "validation_passed_degraded_only")
    )

    assert decision.scheme == "degraded_backfill"


def test_proof_true_strict_false_degraded_false_selects_block() -> None:
    decision = decide_backfill_scheme(
        proof(True), validation(False, False, "weight_sum_violation")
    )

    assert decision.scheme == "block"
    assert decision.reason == "block_weight_sum_violation"


def test_proof_false_strict_false_degraded_false_selects_block() -> None:
    decision = decide_backfill_scheme(
        proof(False), validation(False, False, "insufficient_join_coverage")
    )

    assert decision.scheme == "block"
    assert decision.reason == "block_insufficient_join_coverage"
    assert decision.scheme in {"strict_recovery", "degraded_backfill", "block"}
