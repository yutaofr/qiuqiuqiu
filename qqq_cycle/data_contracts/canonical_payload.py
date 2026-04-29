"""Deterministic canonical payload hashing for structured holdings payloads.

Inputs:
    JSON payload bytes or parsed JSON-like structured data.
Outputs:
    Canonical text plus SHA-256 over the canonical text.
Time semantics:
    Pure content transformation only. No timestamps are inferred.
As-of semantics:
    Canonicalization preserves structured payload semantics and uses Decimal for
    numeric normalization to avoid binary-float hash drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from typing import Any

import json


CANONICALIZATION_METHOD = "json_ast_decimal/v1"
CANONICALIZATION_VERSION = "1"


class CanonicalPayloadError(ValueError):
    """Raised when canonicalization cannot safely preserve payload semantics."""


@dataclass(frozen=True)
class CanonicalPayloadResult:
    canonicalization_method: str
    canonicalization_version: str
    canonical_payload_text: str
    canonical_content_sha256: str
    canonical_payload_size: int
    canonical_row_count: int | None
    canonical_weight_sum: str | None
    canonical_field_set: list[str]
    canonicalization_diagnostics: dict[str, Any]
    numeric_canonicalization_uses_decimal: bool


def _loads_json_payload(raw_payload: bytes | str) -> Any:
    text = raw_payload.decode("utf-8") if isinstance(raw_payload, bytes) else raw_payload
    try:
        return json.loads(text, parse_float=Decimal, parse_int=Decimal)
    except (json.JSONDecodeError, UnicodeDecodeError, InvalidOperation) as exc:
        raise CanonicalPayloadError("payload is not canonicalizable JSON") from exc


def _format_decimal(value: Decimal) -> str:
    if value.is_zero():
        return "0"
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    return text


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, Decimal):
        return _format_decimal(value)
    return value


def _field_set(value: Any) -> set[str]:
    fields: set[str] = set()
    if isinstance(value, dict):
        fields.update(value.keys())
        for nested in value.values():
            fields.update(_field_set(nested))
    elif isinstance(value, list):
        for item in value:
            fields.update(_field_set(item))
    return fields


def _holdings_rows(value: Any) -> tuple[int | None, str | None]:
    if not isinstance(value, dict):
        return None, None
    holdings = value.get("holdings")
    if not isinstance(holdings, list):
        return None, None
    weight_sum = Decimal("0")
    for row in holdings:
        if isinstance(row, dict) and "weight" in row:
            row_weight = row["weight"]
            if isinstance(row_weight, Decimal):
                weight_sum += row_weight
            else:
                weight_sum += Decimal(str(row_weight))
    return len(holdings), _format_decimal(weight_sum)


def canonicalize_json_payload(raw_payload: bytes | str | dict[str, Any] | list[Any]) -> CanonicalPayloadResult:
    parsed = _loads_json_payload(raw_payload) if isinstance(raw_payload, (bytes, str)) else raw_payload
    row_count, weight_sum = _holdings_rows(parsed)
    canonical_obj = _canonicalize(parsed)
    canonical_text = json.dumps(
        canonical_obj,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=False,
    )
    canonical_hash = sha256(canonical_text.encode("utf-8")).hexdigest()
    return CanonicalPayloadResult(
        canonicalization_method=CANONICALIZATION_METHOD,
        canonicalization_version=CANONICALIZATION_VERSION,
        canonical_payload_text=canonical_text,
        canonical_content_sha256=canonical_hash,
        canonical_payload_size=len(canonical_text.encode("utf-8")),
        canonical_row_count=row_count,
        canonical_weight_sum=weight_sum,
        canonical_field_set=sorted(_field_set(parsed)),
        canonicalization_diagnostics={"top_level_type": type(parsed).__name__},
        numeric_canonicalization_uses_decimal=True,
    )


def compute_canonical_sha256(raw_payload: bytes | str | dict[str, Any] | list[Any]) -> str:
    return canonicalize_json_payload(raw_payload).canonical_content_sha256
