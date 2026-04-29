from __future__ import annotations

from qqq_cycle.data_contracts.publication_proof import (
    PublicationProof,
    evaluate_publication_proof,
    sha256_bytes,
)


RAW = b"official holdings payload"
HASH = sha256_bytes(RAW)
SLA = "2026-04-25T16:00:00Z"
SOURCE = "https://www.invesco.com/us/financial-products/etfs/holdings?ticker=QQQ"


def proof(**overrides: object) -> PublicationProof:
    base = {
        "source_label": "Invesco official QQQ holdings export",
        "source_url": SOURCE,
        "content_sha256": HASH,
        "fetched_at_utc": "2026-04-25T15:00:00Z",
        "evidence_class": "direct_http_capture_at_or_before_sla",
        "evidence_timestamp_utc": "2026-04-25T15:00:00Z",
        "http_status": 200,
        "http_date_header": "Sat, 25 Apr 2026 15:00:00 GMT",
        "etag": None,
        "last_modified_header": None,
        "object_version_id": None,
        "audit_log_sha256": None,
        "third_party_snapshot_url": None,
        "strict_eligible": False,
        "strict_eligibility_reason": "missing_machine_evidence",
    }
    base.update(overrides)
    return PublicationProof.from_mapping(base)


def test_direct_http_capture_before_sla_is_strict_eligible() -> None:
    result = evaluate_publication_proof(proof(), SLA, raw_payload=RAW)

    assert result.strict_eligible is True
    assert result.strict_eligibility_reason == "verified_direct_http_capture_before_sla"


def test_direct_http_capture_after_sla_is_not_strict_eligible() -> None:
    result = evaluate_publication_proof(
        proof(evidence_timestamp_utc="2026-04-25T16:00:01Z"), SLA, raw_payload=RAW
    )

    assert result.strict_eligible is False
    assert result.strict_eligibility_reason == "evidence_after_sla_cutoff"


def test_wayback_timestamp_before_sla_is_parsed_and_eligible() -> None:
    result = evaluate_publication_proof(
        proof(evidence_timestamp_utc="20260425155959"),
        SLA,
        raw_payload=RAW,
    )

    assert result.strict_eligible is True
    assert result.strict_eligibility_reason == "verified_direct_http_capture_before_sla"


def test_wayback_timestamp_after_sla_is_rejected() -> None:
    result = evaluate_publication_proof(
        proof(evidence_timestamp_utc="20260425160001"),
        SLA,
        raw_payload=RAW,
    )

    assert result.strict_eligible is False
    assert result.strict_eligibility_reason == "evidence_after_sla_cutoff"


def test_missing_evidence_timestamp_fails_strict() -> None:
    result = evaluate_publication_proof(proof(evidence_timestamp_utc=None), SLA, raw_payload=RAW)

    assert result.strict_eligible is False
    assert result.strict_eligibility_reason == "missing_machine_evidence"


def test_hash_mismatch_fails_strict() -> None:
    result = evaluate_publication_proof(proof(content_sha256=HASH), SLA, raw_payload=b"different")

    assert result.strict_eligible is False
    assert result.strict_eligibility_reason == "hash_mismatch"


def test_simulated_field_present_fails_strict() -> None:
    value = proof().to_dict()
    value["simulated_publication_timestamp"] = "2026-04-25T15:00:00Z"

    result = evaluate_publication_proof(value, SLA, raw_payload=RAW)

    assert result.strict_eligible is False
    assert result.strict_eligibility_reason == "simulated_field_detected"


def test_http_200_with_portfolio_asof_only_fails_strict() -> None:
    result = evaluate_publication_proof(
        proof(evidence_class="portfolio_as_of_date", evidence_timestamp_utc="2026-04-24T00:00:00Z"),
        SLA,
        raw_payload=RAW,
    )

    assert result.strict_eligible is False
    assert result.strict_eligibility_reason == "missing_machine_evidence"


def test_etag_only_fails_strict() -> None:
    result = evaluate_publication_proof(proof(evidence_timestamp_utc=None, etag='"abc"'), SLA, raw_payload=RAW)

    assert result.strict_eligible is False
    assert result.strict_eligibility_reason == "missing_machine_evidence"


def test_last_modified_only_after_cutoff_fails_strict() -> None:
    result = evaluate_publication_proof(
        proof(
            evidence_timestamp_utc="2026-04-25T17:00:00Z",
            last_modified_header="Sat, 25 Apr 2026 17:00:00 GMT",
        ),
        SLA,
        raw_payload=RAW,
    )

    assert result.strict_eligible is False
    assert result.strict_eligibility_reason == "evidence_after_sla_cutoff"


def test_prefilled_strict_eligible_is_ignored_by_evaluator() -> None:
    result = evaluate_publication_proof(
        proof(strict_eligible=True, http_status=404, strict_eligibility_reason="verified_direct_http_capture_before_sla"),
        SLA,
        raw_payload=RAW,
    )

    assert result.strict_eligible is False
    assert result.strict_eligibility_reason == "http_status_not_success"


def test_missing_sla_cutoff_fails_strict() -> None:
    result = evaluate_publication_proof(proof(), None, raw_payload=RAW)

    assert result.strict_eligible is False
    assert result.strict_eligibility_reason == "missing_sla_cutoff"


def test_trusted_third_party_snapshot_with_exact_hash_before_sla_is_eligible() -> None:
    result = evaluate_publication_proof(
        proof(
            source_url="https://webcache.example/qqq",
            evidence_class="trusted_third_party_snapshot_at_or_before_sla",
            third_party_snapshot_url="https://archive.example/snapshot/123",
            verifier_sha256="abc123",
            payload_extraction_method="identity_bytes/v1",
            http_status=None,
        ),
        SLA,
        raw_payload=RAW,
    )

    assert result.strict_eligible is True
    assert result.strict_eligibility_reason == "verified_third_party_snapshot_before_sla"


def test_trusted_third_party_snapshot_with_canonical_hash_before_sla_is_eligible() -> None:
    result = evaluate_publication_proof(
        proof(
            source_url="https://webcache.example/qqq",
            content_sha256="not-the-raw-hash",
            evidence_class="trusted_third_party_snapshot_at_or_before_sla",
            third_party_snapshot_url="https://archive.example/snapshot/123",
            canonical_content_sha256="772be1eacd8b93529b37c94ba64fc119e01b20610c75779de32f5c573f501072",
            canonicalization_method="json_ast_decimal/v1",
            canonicalization_version="1",
            verifier_sha256="abc123",
            payload_extraction_method="identity_bytes/v1",
            http_status=None,
        ),
        SLA,
        raw_payload=b'{"holdings":[{"ticker":"AAA","weight":0.4}]}',
    )

    assert result.strict_eligible is True
    assert result.strict_eligibility_reason == "verified_third_party_snapshot_before_sla"


def test_trusted_third_party_snapshot_missing_extraction_binding_fails() -> None:
    result = evaluate_publication_proof(
        proof(
            source_url="https://webcache.example/qqq",
            evidence_class="trusted_third_party_snapshot_at_or_before_sla",
            third_party_snapshot_url="https://archive.example/snapshot/123",
            verifier_sha256=None,
            payload_extraction_method=None,
            http_status=None,
        ),
        SLA,
        raw_payload=RAW,
    )

    assert result.strict_eligible is False
    assert result.strict_eligibility_reason == "missing_machine_evidence"


def test_trusted_third_party_snapshot_with_unknown_canonicalization_fails() -> None:
    result = evaluate_publication_proof(
        proof(
            source_url="https://webcache.example/qqq",
            content_sha256="not-the-raw-hash",
            evidence_class="trusted_third_party_snapshot_at_or_before_sla",
            third_party_snapshot_url="https://archive.example/snapshot/123",
            canonical_content_sha256="772be1eacd8b93529b37c94ba64fc119e01b20610c75779de32f5c573f501072",
            canonicalization_method="unknown/v9",
            canonicalization_version="9",
            verifier_sha256="abc123",
            payload_extraction_method="identity_bytes/v1",
            http_status=None,
        ),
        SLA,
        raw_payload=b'{"holdings":[{"ticker":"AAA","weight":0.4}]}',
    )

    assert result.strict_eligible is False
    assert result.strict_eligibility_reason == "missing_machine_evidence"
