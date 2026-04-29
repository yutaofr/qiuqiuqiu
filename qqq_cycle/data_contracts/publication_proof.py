"""Publication proof contract for controlled QQQ backfill ingest.

Inputs:
    Raw captured bytes from an official source and a proof mapping.
Outputs:
    A PublicationProof with strict eligibility derived by machine logic.
Time semantics:
    Strict eligibility is evaluated against an explicit UTC SLA cutoff. The
    portfolio as-of date, filename date, local file timestamp, and current
    runtime are not publication evidence.
As-of semantics:
    A proof is strict-eligible only when machine evidence shows the captured
    content was available at or before the operational SLA cutoff.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

import json

from qqq_cycle.data_contracts.canonical_payload import (
    CANONICALIZATION_METHOD,
    compute_canonical_sha256,
)


STRICT_EVIDENCE_CLASSES = frozenset(
    {
        "direct_http_capture_at_or_before_sla",
        "immutable_audit_log_at_or_before_sla",
        "trusted_object_version_at_or_before_sla",
        "trusted_third_party_snapshot_at_or_before_sla",
    }
)

STRICT_ELIGIBILITY_REASONS = frozenset(
    {
        "verified_direct_http_capture_before_sla",
        "verified_immutable_audit_log_before_sla",
        "verified_object_version_before_sla",
        "verified_third_party_snapshot_before_sla",
        "missing_machine_evidence",
        "evidence_after_sla_cutoff",
        "hash_mismatch",
        "source_not_allowed",
        "http_status_not_success",
        "simulated_field_detected",
        "missing_sla_cutoff",
    }
)

FORBIDDEN_FIELD_FRAGMENTS = (
    "simulated",
    "mocked",
    "synthetic_timestamp",
    "assumed_publication",
    "manual_publication",
    "operator_asserted",
)

ALLOWED_INVESCO_HOSTS = frozenset({"invesco.com", "www.invesco.com"})


@dataclass(frozen=True)
class PublicationProof:
    source_label: str
    source_url: str
    content_sha256: str
    fetched_at_utc: str
    evidence_class: str
    evidence_timestamp_utc: str | None
    http_status: int | None
    http_date_header: str | None
    etag: str | None
    last_modified_header: str | None
    object_version_id: str | None
    audit_log_sha256: str | None
    third_party_snapshot_url: str | None
    strict_eligible: bool = False
    strict_eligibility_reason: str = "missing_machine_evidence"
    canonical_content_sha256: str | None = None
    canonicalization_method: str | None = None
    canonicalization_version: str | None = None
    payload_extraction_method: str | None = None
    verifier_sha256: str | None = None
    raw_payload_hash_match: bool | None = None
    canonical_payload_hash_match: bool | None = None
    wrapper_detected: bool | None = None
    wrapper_removed: bool | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PublicationProof":
        return cls(
            source_label=str(value.get("source_label", "")),
            source_url=str(value.get("source_url", "")),
            content_sha256=str(value.get("content_sha256", "")),
            fetched_at_utc=str(value.get("fetched_at_utc", "")),
            evidence_class=str(value.get("evidence_class", "")),
            evidence_timestamp_utc=value.get("evidence_timestamp_utc"),
            http_status=value.get("http_status"),
            http_date_header=value.get("http_date_header"),
            etag=value.get("etag"),
            last_modified_header=value.get("last_modified_header"),
            object_version_id=value.get("object_version_id"),
            audit_log_sha256=value.get("audit_log_sha256"),
            third_party_snapshot_url=value.get("third_party_snapshot_url"),
            canonical_content_sha256=value.get("canonical_content_sha256"),
            canonicalization_method=value.get("canonicalization_method"),
            canonicalization_version=value.get("canonicalization_version"),
            payload_extraction_method=value.get("payload_extraction_method"),
            verifier_sha256=value.get("verifier_sha256"),
            raw_payload_hash_match=value.get("raw_payload_hash_match"),
            canonical_payload_hash_match=value.get("canonical_payload_hash_match"),
            wrapper_detected=value.get("wrapper_detected"),
            wrapper_removed=value.get("wrapper_removed"),
            strict_eligible=bool(value.get("strict_eligible", False)),
            strict_eligibility_reason=str(
                value.get("strict_eligibility_reason", "missing_machine_evidence")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_bytes(raw_payload: bytes) -> str:
    return sha256(raw_payload).hexdigest()


def contains_forbidden_proof_field(proof_mapping: Mapping[str, Any]) -> bool:
    for key in proof_mapping:
        lowered = str(key).lower()
        if any(fragment in lowered for fragment in FORBIDDEN_FIELD_FRAGMENTS):
            return True
    return False


def parse_utc_timestamp(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def source_url_allowed(source_url: str, allowed_hosts: Iterable[str] = ALLOWED_INVESCO_HOSTS) -> bool:
    host = urlparse(source_url).hostname
    if host is None:
        return False
    host = host.lower()
    allowed = {item.lower() for item in allowed_hosts}
    return host in allowed or any(host.endswith("." + item) for item in allowed)


def trusted_immutable_snapshot_exists(proof: PublicationProof) -> bool:
    if proof.evidence_class == "immutable_audit_log_at_or_before_sla":
        return bool(proof.audit_log_sha256)
    if proof.evidence_class == "trusted_object_version_at_or_before_sla":
        return bool(proof.object_version_id)
    if proof.evidence_class == "trusted_third_party_snapshot_at_or_before_sla":
        return bool(proof.third_party_snapshot_url)
    return False


def reason_for_verified_class(evidence_class: str) -> str:
    if evidence_class == "direct_http_capture_at_or_before_sla":
        return "verified_direct_http_capture_before_sla"
    if evidence_class == "immutable_audit_log_at_or_before_sla":
        return "verified_immutable_audit_log_before_sla"
    if evidence_class == "trusted_object_version_at_or_before_sla":
        return "verified_object_version_before_sla"
    if evidence_class == "trusted_third_party_snapshot_at_or_before_sla":
        return "verified_third_party_snapshot_before_sla"
    return "missing_machine_evidence"


def evaluate_publication_proof(
    proof: PublicationProof | Mapping[str, Any],
    sla_cutoff_utc: str | datetime | None,
    *,
    raw_payload: bytes | None = None,
    allowed_source_hosts: Iterable[str] = ALLOWED_INVESCO_HOSTS,
) -> PublicationProof:
    """Derive strict eligibility from machine evidence and an explicit SLA cutoff."""

    proof_mapping = proof if isinstance(proof, Mapping) else proof.to_dict()
    parsed = PublicationProof.from_mapping(proof_mapping)

    def rejected(reason: str) -> PublicationProof:
        if reason not in STRICT_ELIGIBILITY_REASONS:
            raise ValueError(f"unknown strict eligibility reason: {reason}")
        return replace(parsed, strict_eligible=False, strict_eligibility_reason=reason)

    if contains_forbidden_proof_field(proof_mapping):
        return rejected("simulated_field_detected")

    cutoff = parse_utc_timestamp(sla_cutoff_utc)
    if cutoff is None:
        return rejected("missing_sla_cutoff")

    if not parsed.content_sha256:
        return rejected("missing_machine_evidence")
    raw_hash_match = raw_payload is not None and sha256_bytes(raw_payload) == parsed.content_sha256
    canonical_hash_match = False
    if parsed.canonical_content_sha256:
        if (
            parsed.canonicalization_method != CANONICALIZATION_METHOD
            or not parsed.canonicalization_version
        ):
            return rejected("missing_machine_evidence")
        if raw_payload is not None:
            canonical_hash_match = compute_canonical_sha256(raw_payload) == parsed.canonical_content_sha256
    if raw_payload is not None and not raw_hash_match and not canonical_hash_match:
        return rejected("hash_mismatch")

    evidence_ts = parse_utc_timestamp(parsed.evidence_timestamp_utc)
    if evidence_ts is None:
        return rejected("missing_machine_evidence")
    if evidence_ts > cutoff:
        return rejected("evidence_after_sla_cutoff")

    if parsed.evidence_class not in STRICT_EVIDENCE_CLASSES:
        return rejected("missing_machine_evidence")

    third_party_proof = parsed.evidence_class == "trusted_third_party_snapshot_at_or_before_sla"
    if not third_party_proof and not source_url_allowed(parsed.source_url, allowed_source_hosts):
        return rejected("source_not_allowed")

    if parsed.evidence_class == "direct_http_capture_at_or_before_sla":
        if parsed.http_status != 200:
            return rejected("http_status_not_success")
    elif third_party_proof:
        if (
            not parsed.third_party_snapshot_url
            or not parsed.verifier_sha256
            or not parsed.payload_extraction_method
        ):
            return rejected("missing_machine_evidence")
        if parsed.wrapper_detected and not parsed.wrapper_removed:
            return rejected("missing_machine_evidence")
    elif not trusted_immutable_snapshot_exists(parsed):
        return rejected("missing_machine_evidence")

    return replace(
        parsed,
        raw_payload_hash_match=raw_hash_match,
        canonical_payload_hash_match=canonical_hash_match,
        strict_eligible=True,
        strict_eligibility_reason=reason_for_verified_class(parsed.evidence_class),
    )


def write_publication_proof(path: Path, proof: PublicationProof) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(proof.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_publication_proof(path: Path) -> PublicationProof:
    return PublicationProof.from_mapping(json.loads(path.read_text(encoding="utf-8")))
