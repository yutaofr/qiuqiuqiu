"""Strict evidence helpers for immutable publication-proof upgrade attempts.

Inputs:
    Candidate snapshot bytes plus the current official raw payload bytes.
Outputs:
    Wrapper detection, extracted payload bytes, and exact/canonical hash
    equivalence diagnostics.
Time semantics:
    No timestamps are manufactured; these helpers operate on supplied content.
As-of semantics:
    Hash equivalence is proven only from extracted structured payload bytes or a
    deterministic canonical form built from structured data.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import base64
import json

from qqq_cycle.data_contracts.canonical_payload import (
    CANONICALIZATION_METHOD,
    CANONICALIZATION_VERSION,
    CanonicalPayloadError,
    compute_canonical_sha256,
)


@dataclass(frozen=True)
class SnapshotWrapperDetection:
    wrapper_detected: bool
    wrapper_kind: str
    payload_extraction_method: str


@dataclass(frozen=True)
class ExtractedPayload:
    extracted_payload: bytes
    wrapper_removed: bool
    payload_extraction_method: str


@dataclass(frozen=True)
class PayloadEquivalenceResult:
    raw_payload_hash_match: bool
    canonical_payload_hash_match: bool
    content_sha256: str
    canonical_content_sha256: str | None
    payload_extraction_method: str
    wrapper_detected: bool
    wrapper_removed: bool
    canonicalization_method: str | None
    canonicalization_version: str | None
    numeric_canonicalization_uses_decimal: bool


def _looks_like_json(raw_payload: bytes) -> bool:
    text = raw_payload.lstrip()[:1]
    return text in {b"{", b"["}


def detect_snapshot_wrapper(raw_payload: bytes, content_type: str | None = None) -> SnapshotWrapperDetection:
    lowered = (content_type or "").lower()
    if "json" in lowered and _looks_like_json(raw_payload):
        try:
            parsed = json.loads(raw_payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return SnapshotWrapperDetection(True, "unparseable_json_wrapper", "unextractable_wrapper/v1")
        if isinstance(parsed, dict) and isinstance(parsed.get("archived_payload_base64"), str):
            return SnapshotWrapperDetection(True, "structured_snapshot_envelope_json", "structured_snapshot_envelope_json/v1")
        return SnapshotWrapperDetection(False, "none", "identity_bytes/v1")
    if _looks_like_json(raw_payload):
        return SnapshotWrapperDetection(False, "none", "identity_bytes/v1")
    return SnapshotWrapperDetection(True, "opaque_wrapper", "unextractable_wrapper/v1")


def extract_original_payload(
    raw_payload: bytes,
    detection: SnapshotWrapperDetection,
) -> ExtractedPayload:
    if not detection.wrapper_detected:
        return ExtractedPayload(
            extracted_payload=raw_payload,
            wrapper_removed=False,
            payload_extraction_method=detection.payload_extraction_method,
        )
    if detection.payload_extraction_method == "structured_snapshot_envelope_json/v1":
        parsed = json.loads(raw_payload.decode("utf-8"))
        payload_text = parsed["archived_payload_base64"]
        return ExtractedPayload(
            extracted_payload=base64.b64decode(payload_text),
            wrapper_removed=True,
            payload_extraction_method=detection.payload_extraction_method,
        )
    raise ValueError("snapshot wrapper detected but extraction method is not supported")


def compute_raw_sha256(raw_payload: bytes) -> str:
    return sha256(raw_payload).hexdigest()


def compute_payload_sha256(raw_payload: bytes, content_type: str | None = None) -> tuple[str, SnapshotWrapperDetection, ExtractedPayload]:
    detection = detect_snapshot_wrapper(raw_payload, content_type)
    extraction = extract_original_payload(raw_payload, detection)
    return compute_raw_sha256(extraction.extracted_payload), detection, extraction


def compute_canonical_sha256_for_payload(raw_payload: bytes) -> str:
    return compute_canonical_sha256(raw_payload)


def compute_payload_equivalence(
    *,
    candidate_payload: bytes,
    current_raw_payload: bytes,
    content_type: str | None = "application/json",
) -> PayloadEquivalenceResult:
    candidate_sha256, detection, extraction = compute_payload_sha256(candidate_payload, content_type)
    current_sha256 = compute_raw_sha256(current_raw_payload)
    raw_match = candidate_sha256 == current_sha256

    canonical_candidate_sha: str | None = None
    canonical_current_sha: str | None = None
    canonical_match = False
    try:
        canonical_candidate_sha = compute_canonical_sha256_for_payload(extraction.extracted_payload)
        canonical_current_sha = compute_canonical_sha256_for_payload(current_raw_payload)
        canonical_match = canonical_candidate_sha == canonical_current_sha
    except CanonicalPayloadError:
        canonical_match = False

    return PayloadEquivalenceResult(
        raw_payload_hash_match=raw_match,
        canonical_payload_hash_match=canonical_match,
        content_sha256=candidate_sha256,
        canonical_content_sha256=canonical_candidate_sha if canonical_match else None,
        payload_extraction_method=extraction.payload_extraction_method,
        wrapper_detected=detection.wrapper_detected,
        wrapper_removed=extraction.wrapper_removed,
        canonicalization_method=CANONICALIZATION_METHOD if canonical_match else None,
        canonicalization_version=CANONICALIZATION_VERSION if canonical_match else None,
        numeric_canonicalization_uses_decimal=True,
    )
