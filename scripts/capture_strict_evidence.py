#!/usr/bin/env python3
"""Enumerate and evaluate strict-evidence upgrade candidates for QQQ holdings.

This script inventories candidate machine evidence, writes a candidate strict
proof artifact, and records a machine evaluation outcome. It never upgrades the
accepted scheme on its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qqq_cycle.data_contracts.publication_proof import (
    PublicationProof,
    evaluate_publication_proof,
    read_publication_proof,
    write_publication_proof,
)
from qqq_cycle.data_contracts.strict_evidence import compute_payload_equivalence


HTML_HOLDINGS_URL = (
    "https://www.invesco.com/us/financial-products/etfs/holdings?audienceType=Investor&ticker=QQQ"
)


@dataclass(frozen=True)
class InventoryEntry:
    evidence_source: str
    evidence_class_candidate: str
    available: bool
    timestamp_utc: str | None
    content_hash_binding: str
    canonical_hash_binding: str
    source_url_binding: str
    immutable_or_mutable: str
    machine_verifiable: bool
    wrapper_risk: str
    numeric_serialization_risk: str
    risk: str
    decision: str


def _json_get(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "qqq-cycle-strict-evidence/1.0"})
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _availability_query(source_url: str, timestamp: str) -> dict[str, Any]:
    query = urlencode({"url": source_url, "timestamp": timestamp})
    return _json_get(f"https://archive.org/wayback/available?{query}")


def _archive_entry(source_url: str, timestamp: str, current_raw_payload: bytes) -> tuple[InventoryEntry, dict[str, Any] | None]:
    try:
        payload = _availability_query(source_url, timestamp)
    except Exception as exc:  # pragma: no cover - network diagnostics
        return (
            InventoryEntry(
                evidence_source=f"archive.org availability {source_url}",
                evidence_class_candidate="trusted_third_party_snapshot_at_or_before_sla",
                available=False,
                timestamp_utc=None,
                content_hash_binding="none",
                canonical_hash_binding="none",
                source_url_binding="query_only",
                immutable_or_mutable="immutable",
                machine_verifiable=True,
                wrapper_risk="unknown",
                numeric_serialization_risk="unknown",
                risk=f"archive query failed: {type(exc).__name__}",
                decision="reject_no_machine_snapshot_result",
            ),
            None,
        )

    archived = payload.get("archived_snapshots", {})
    closest = archived.get("closest") if isinstance(archived, dict) else None
    if not isinstance(closest, dict):
        return (
            InventoryEntry(
                evidence_source=f"archive.org availability {source_url}",
                evidence_class_candidate="trusted_third_party_snapshot_at_or_before_sla",
                available=False,
                timestamp_utc=None,
                content_hash_binding="none",
                canonical_hash_binding="none",
                source_url_binding="query_only",
                immutable_or_mutable="immutable",
                machine_verifiable=True,
                wrapper_risk="unknown",
                numeric_serialization_risk="unknown",
                risk="no archived snapshot returned before SLA query timestamp",
                decision="reject_no_snapshot_before_sla",
            ),
            None,
        )

    snapshot_url = str(closest.get("url", ""))
    snapshot_timestamp = str(closest.get("timestamp", ""))
    try:
        request = Request(snapshot_url, headers={"User-Agent": "qqq-cycle-strict-evidence/1.0"})
        with urlopen(request, timeout=20) as response:
            snapshot_bytes = response.read()
            content_type = response.headers.get("Content-Type")
    except Exception as exc:  # pragma: no cover - network diagnostics
        return (
            InventoryEntry(
                evidence_source=f"archive.org snapshot {snapshot_url}",
                evidence_class_candidate="trusted_third_party_snapshot_at_or_before_sla",
                available=True,
                timestamp_utc=snapshot_timestamp,
                content_hash_binding="unverified",
                canonical_hash_binding="unverified",
                source_url_binding="snapshot_url_present",
                immutable_or_mutable="immutable",
                machine_verifiable=True,
                wrapper_risk="unknown",
                numeric_serialization_risk="unknown",
                risk=f"snapshot fetch failed: {type(exc).__name__}",
                decision="reject_snapshot_fetch_failed",
            ),
            None,
        )

    equivalence = compute_payload_equivalence(
        candidate_payload=snapshot_bytes,
        current_raw_payload=current_raw_payload,
        content_type=content_type,
    )
    exact = equivalence.raw_payload_hash_match
    canonical = equivalence.canonical_payload_hash_match
    decision = "candidate_for_strict_evaluation" if (exact or canonical) else "reject_hash_not_equivalent"
    return (
        InventoryEntry(
            evidence_source=f"archive.org snapshot {snapshot_url}",
            evidence_class_candidate="trusted_third_party_snapshot_at_or_before_sla",
            available=True,
            timestamp_utc=snapshot_timestamp,
            content_hash_binding="exact_payload_sha256" if exact else "none",
            canonical_hash_binding="canonical_ast_sha256" if canonical else "none",
            source_url_binding="snapshot_url_present",
            immutable_or_mutable="immutable",
            machine_verifiable=True,
            wrapper_risk="detected" if equivalence.wrapper_detected else "none",
            numeric_serialization_risk="controlled_decimal" if canonical else "none",
            risk="none" if decision == "candidate_for_strict_evaluation" else "hash equivalence failed",
            decision=decision,
        ),
        {
            "evidence_class": "trusted_third_party_snapshot_at_or_before_sla",
            "evidence_timestamp_utc": snapshot_timestamp,
            "third_party_snapshot_url": snapshot_url,
            "content_sha256": equivalence.content_sha256,
            "canonical_content_sha256": equivalence.canonical_content_sha256,
            "payload_extraction_method": equivalence.payload_extraction_method,
            "wrapper_detected": equivalence.wrapper_detected,
            "wrapper_removed": equivalence.wrapper_removed,
            "canonicalization_method": equivalence.canonicalization_method,
            "canonicalization_version": equivalence.canonicalization_version,
            "verifier_sha256": equivalence.content_sha256,
            "raw_payload_hash_match": equivalence.raw_payload_hash_match,
            "canonical_payload_hash_match": equivalence.canonical_payload_hash_match,
            "numeric_canonicalization_uses_decimal": equivalence.numeric_canonicalization_uses_decimal,
        }
        if decision == "candidate_for_strict_evaluation"
        else None,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week-end", default="2026-04-24")
    parser.add_argument("--raw-path", type=Path, required=True)
    parser.add_argument("--proof-path", type=Path, required=True)
    parser.add_argument("--sla-cutoff-utc", required=True)
    parser.add_argument("--inventory-output", type=Path, default=None)
    parser.add_argument("--candidate-proof-output", type=Path, default=None)
    parser.add_argument("--evaluation-output", type=Path, default=None)
    args = parser.parse_args()

    raw_payload = args.raw_path.read_bytes()
    original_proof = read_publication_proof(args.proof_path)
    inventory_output = args.inventory_output or Path("outputs/phase14") / f"strict_evidence_inventory_{args.week_end}.json"
    candidate_output = args.candidate_proof_output or Path("captures") / f"invesco_qqq_holdings_{args.week_end}_proof_candidate_strict.json"
    evaluation_output = args.evaluation_output or Path("outputs/phase14") / f"strict_upgrade_evaluation_{args.week_end}.json"

    inventory: list[InventoryEntry] = [
        InventoryEntry(
            evidence_source="local current proof artifact",
            evidence_class_candidate=original_proof.evidence_class,
            available=True,
            timestamp_utc=original_proof.evidence_timestamp_utc,
            content_hash_binding="exact_payload_sha256",
            canonical_hash_binding="none",
            source_url_binding="proof_source_url_present" if original_proof.source_url else "none",
            immutable_or_mutable="mutable_local_artifact",
            machine_verifiable=True,
            wrapper_risk="none",
            numeric_serialization_risk="none",
            risk="evidence timestamp after SLA cutoff",
            decision="reject_after_sla",
        ),
        InventoryEntry(
            evidence_source="local raw capture file",
            evidence_class_candidate="direct_http_capture_at_or_before_sla",
            available=True,
            timestamp_utc=None,
            content_hash_binding="exact_payload_sha256",
            canonical_hash_binding="none",
            source_url_binding="none",
            immutable_or_mutable="mutable_local_artifact",
            machine_verifiable=False,
            wrapper_risk="none",
            numeric_serialization_risk="none",
            risk="local file alone is not publication availability proof",
            decision="reject_no_publication_timestamp_binding",
        ),
        InventoryEntry(
            evidence_source="internal immutable audit log",
            evidence_class_candidate="immutable_audit_log_at_or_before_sla",
            available=False,
            timestamp_utc=None,
            content_hash_binding="none",
            canonical_hash_binding="none",
            source_url_binding="none",
            immutable_or_mutable="unknown",
            machine_verifiable=False,
            wrapper_risk="none",
            numeric_serialization_risk="none",
            risk="no append-only audit log artifact present in repo",
            decision="reject_missing_source",
        ),
        InventoryEntry(
            evidence_source="trusted object version metadata",
            evidence_class_candidate="trusted_object_version_at_or_before_sla",
            available=False,
            timestamp_utc=None,
            content_hash_binding="none",
            canonical_hash_binding="none",
            source_url_binding="none",
            immutable_or_mutable="unknown",
            machine_verifiable=False,
            wrapper_risk="none",
            numeric_serialization_risk="none",
            risk="no object version metadata present in repo",
            decision="reject_missing_source",
        ),
    ]

    archive_timestamp = args.sla_cutoff_utc.replace("-", "").replace(":", "").replace("T", "").replace("Z", "")
    candidate_updates: dict[str, Any] | None = None
    for source_url in (original_proof.source_url, HTML_HOLDINGS_URL):
        entry, updates = _archive_entry(source_url, archive_timestamp, raw_payload)
        inventory.append(entry)
        if updates is not None and candidate_updates is None:
            candidate_updates = updates

    _write_json(
        inventory_output,
        {
            "week_end": args.week_end,
            "sla_cutoff_utc": args.sla_cutoff_utc,
            "inventory": [asdict(entry) for entry in inventory],
        },
    )

    candidate_mapping = original_proof.to_dict()
    if candidate_updates is not None:
        candidate_mapping.update(candidate_updates)
        candidate_mapping["source_url"] = original_proof.source_url
    candidate_proof = PublicationProof.from_mapping(candidate_mapping)
    evaluated_candidate = evaluate_publication_proof(
        candidate_proof,
        args.sla_cutoff_utc,
        raw_payload=raw_payload,
    )
    write_publication_proof(candidate_output, evaluated_candidate)

    _write_json(
        evaluation_output,
        {
            "week_end": args.week_end,
            "original_strict_eligible": original_proof.strict_eligible,
            "candidate_strict_eligible": evaluated_candidate.strict_eligible,
            "original_reason": original_proof.strict_eligibility_reason,
            "candidate_reason": evaluated_candidate.strict_eligibility_reason,
            "evidence_class": evaluated_candidate.evidence_class,
            "evidence_timestamp_utc": evaluated_candidate.evidence_timestamp_utc,
            "sla_cutoff_utc": args.sla_cutoff_utc,
            "raw_content_sha256_match": bool(evaluated_candidate.raw_payload_hash_match),
            "extracted_payload_hash_match": bool(evaluated_candidate.raw_payload_hash_match),
            "canonical_hash_match": bool(evaluated_candidate.canonical_payload_hash_match),
            "object_version_id": evaluated_candidate.object_version_id,
            "audit_log_sha256": evaluated_candidate.audit_log_sha256,
            "third_party_snapshot_url": evaluated_candidate.third_party_snapshot_url,
            "payload_extraction_method": evaluated_candidate.payload_extraction_method,
            "wrapper_detected": evaluated_candidate.wrapper_detected,
            "wrapper_removed": evaluated_candidate.wrapper_removed,
            "canonicalization_method": evaluated_candidate.canonicalization_method,
            "canonicalization_version": evaluated_candidate.canonicalization_version,
            "numeric_canonicalization_uses_decimal": True,
            "upgrade_allowed": evaluated_candidate.strict_eligible,
            "strict_upgrade_attempted": candidate_updates is not None,
            "strict_upgrade_succeeded": False,
            "strict_recovery_attempted": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
