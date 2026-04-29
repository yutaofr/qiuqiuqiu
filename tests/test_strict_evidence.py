from __future__ import annotations

import base64
import pytest

from qqq_cycle.data_contracts.strict_evidence import (
    compute_payload_equivalence,
    detect_snapshot_wrapper,
    extract_original_payload,
    normalize_wayback_timestamp_to_utc_iso,
)


RAW = b'{"holdings":[{"ticker":"AAA","weight":0.4}]}'


def test_detect_snapshot_wrapper_for_identity_json_payload() -> None:
    detection = detect_snapshot_wrapper(RAW, "application/json")

    assert detection.wrapper_detected is False
    assert detection.payload_extraction_method == "identity_bytes/v1"


def test_extract_original_payload_from_structured_snapshot_envelope() -> None:
    envelope = (
        '{"snapshot_metadata":{"provider":"archive"},"archived_payload_base64":"'
        + base64.b64encode(RAW).decode("ascii")
        + '"}'
    ).encode("utf-8")

    detection = detect_snapshot_wrapper(envelope, "application/json")
    extraction = extract_original_payload(envelope, detection)

    assert detection.wrapper_detected is True
    assert extraction.wrapper_removed is True
    assert extraction.extracted_payload == RAW


def test_compute_payload_equivalence_reports_exact_hash_match() -> None:
    result = compute_payload_equivalence(candidate_payload=RAW, current_raw_payload=RAW)

    assert result.raw_payload_hash_match is True
    assert result.canonical_payload_hash_match is True
    assert result.payload_extraction_method == "identity_bytes/v1"


def test_compute_payload_equivalence_reports_canonical_match_only() -> None:
    candidate_payload = b'{"holdings":[{"ticker":"AAA","weight":0.4000}]}'
    current_payload = b'{"holdings":[{"ticker":"AAA","weight":0.4}]}'

    result = compute_payload_equivalence(
        candidate_payload=candidate_payload,
        current_raw_payload=current_payload,
    )

    assert result.raw_payload_hash_match is False
    assert result.canonical_payload_hash_match is True


def test_normalize_wayback_timestamp_to_utc_iso() -> None:
    assert normalize_wayback_timestamp_to_utc_iso("20260425155959") == "2026-04-25T15:59:59Z"
    assert normalize_wayback_timestamp_to_utc_iso("20260425160001") == "2026-04-25T16:00:01Z"


@pytest.mark.parametrize("value", ["2026042515595", "2026042515595A"])
def test_normalize_wayback_timestamp_rejects_invalid_input(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_wayback_timestamp_to_utc_iso(value)
