from __future__ import annotations

import json

import pytest

from qqq_cycle.data_contracts.canonical_payload import (
    CanonicalPayloadError,
    canonicalize_json_payload,
    compute_canonical_sha256,
)


def test_decimal_text_variants_canonicalize_identically() -> None:
    payload_a = b'{"holdings":[{"ticker":"AAA","weight":0.012}]}'
    payload_b = b'{"holdings":[{"ticker":"AAA","weight":0.0120}]}'
    payload_c = b'{"holdings":[{"ticker":"AAA","weight":1.2e-2}]}'

    result_a = canonicalize_json_payload(payload_a)
    result_b = canonicalize_json_payload(payload_b)
    result_c = canonicalize_json_payload(payload_c)

    assert result_a.canonical_payload_text == result_b.canonical_payload_text
    assert result_a.canonical_payload_text == result_c.canonical_payload_text
    assert result_a.canonical_content_sha256 == result_b.canonical_content_sha256
    assert result_a.canonical_content_sha256 == result_c.canonical_content_sha256


def test_row_deletion_changes_canonical_hash() -> None:
    payload_a = b'{"holdings":[{"ticker":"AAA","weight":0.4},{"ticker":"BBB","weight":0.6}]}'
    payload_b = b'{"holdings":[{"ticker":"AAA","weight":0.4}]}'

    assert compute_canonical_sha256(payload_a) != compute_canonical_sha256(payload_b)


def test_weight_change_changes_canonical_hash() -> None:
    payload_a = b'{"holdings":[{"ticker":"AAA","weight":0.4}]}'
    payload_b = b'{"holdings":[{"ticker":"AAA","weight":0.4001}]}'

    assert compute_canonical_sha256(payload_a) != compute_canonical_sha256(payload_b)


def test_ticker_case_change_changes_canonical_hash() -> None:
    payload_a = b'{"holdings":[{"ticker":"AAA","weight":0.4}]}'
    payload_b = b'{"holdings":[{"ticker":"aaa","weight":0.4}]}'

    assert compute_canonical_sha256(payload_a) != compute_canonical_sha256(payload_b)


def test_wrapper_injected_json_does_not_canonicalize() -> None:
    wrapped = b"wayback-banner\n" + b'{"holdings":[{"ticker":"AAA","weight":0.4}]}'

    with pytest.raises(CanonicalPayloadError):
        canonicalize_json_payload(wrapped)


def test_canonicalization_uses_decimal_not_float() -> None:
    result = canonicalize_json_payload(
        b'{"holdings":[{"ticker":"AAA","weight":1.2e-2},{"ticker":"BBB","weight":-0.0}]}'
    )

    parsed = json.loads(result.canonical_payload_text)
    assert parsed["holdings"][0]["weight"] == "0.012"
    assert parsed["holdings"][1]["weight"] == "0"
    assert result.numeric_canonicalization_uses_decimal is True
