"""Parser contract for official Invesco QQQ holdings payloads.

Inputs:
    Raw captured payload bytes plus content-type/source-url metadata.
Outputs:
    Parsed normalized raw holdings rows ready for canonical namespace mapping.
Time semantics:
    Parser only transforms one captured payload; no runtime data lookups.
As-of semantics:
    Rows and weights are extracted from the captured point-in-time payload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd


class InvescoHoldingsParseError(ValueError):
    """Raised when official holdings payload is absent/incomplete/unparseable."""


@dataclass(frozen=True)
class ParsedHoldings:
    frame: pd.DataFrame
    parser_diagnostics: dict[str, Any]


def _looks_like_html(raw_payload: bytes, content_type: str | None) -> bool:
    if "html" in (content_type or "").lower():
        return True
    prefix = raw_payload[:2048].decode("utf-8", errors="ignore").lower()
    return "<!doctype html" in prefix or "<html" in prefix or "<head" in prefix


def _normalize_weights(raw_weights: pd.Series) -> tuple[pd.Series, str, float]:
    weights = pd.to_numeric(raw_weights, errors="coerce")
    if weights.isna().any():
        raise InvescoHoldingsParseError("weight_unit_unresolved")
    raw_sum = float(weights.sum())
    if 0.99 <= raw_sum <= 1.01:
        return weights.astype(float), "decimal", raw_sum
    if 99.0 <= raw_sum <= 101.0:
        return (weights / 100.0).astype(float), "percentage", raw_sum
    raise InvescoHoldingsParseError("weight_unit_unresolved")


def parse_official_invesco_holdings_payload(
    *,
    raw_payload: bytes,
    source_url: str,
    content_type: str | None = None,
) -> ParsedHoldings:
    """Parse official machine-readable Invesco QQQ holdings payload."""

    if _looks_like_html(raw_payload, content_type):
        raise InvescoHoldingsParseError("official_source_incomplete")

    try:
        payload = json.loads(raw_payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise InvescoHoldingsParseError("official_source_incomplete") from exc

    holdings = None
    effective_date = None
    if isinstance(payload, dict):
        holdings = payload.get("holdings")
        effective_date = payload.get("effectiveDate") or payload.get("effectiveBusinessDate")
    elif isinstance(payload, list):
        holdings = payload
    if not isinstance(holdings, list) or len(holdings) == 0:
        raise InvescoHoldingsParseError("empty_holdings")

    rows: list[dict[str, Any]] = []
    for i, row in enumerate(holdings, start=1):
        if not isinstance(row, dict):
            continue
        raw_symbol = str(row.get("ticker", "") or "").strip()
        raw_name = str(row.get("issuerName", "") or "").strip()
        raw_weight = row.get("percentageOfTotalNetAssets")
        rows.append(
            {
                "source_row_number": i,
                "raw_symbol": raw_symbol,
                "raw_name": raw_name,
                "raw_weight": raw_weight,
                "cusip": str(row.get("cusip", "") or "").strip(),
                "isin": str(row.get("isin", "") or "").strip(),
                "sedol": str(row.get("sedol", "") or "").strip(),
                "exchange": str(row.get("exchange", "NASDAQ") or "").strip(),
                "asset_class": str(row.get("securityTypeName", "equity") or "").strip().lower(),
                "raw_identifier": str(row.get("cusip", "") or "").strip(),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise InvescoHoldingsParseError("empty_holdings")

    normalized_weights, weight_unit, raw_weight_sum = _normalize_weights(frame["raw_weight"])
    frame["raw_weight"] = pd.to_numeric(frame["raw_weight"], errors="coerce")
    frame["normalized_weight"] = normalized_weights
    frame["weight_unit"] = weight_unit

    diagnostics = {
        "source_url": source_url,
        "content_type": content_type,
        "row_count": int(len(frame.index)),
        "raw_weight_sum": raw_weight_sum,
        "normalized_weight_sum": float(frame["normalized_weight"].sum()),
        "weight_unit": weight_unit,
        "effective_date": effective_date,
    }
    return ParsedHoldings(frame=frame, parser_diagnostics=diagnostics)
