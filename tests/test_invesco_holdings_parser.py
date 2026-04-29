from __future__ import annotations

import json

import pytest

from qqq_cycle.data_contracts.invesco_holdings_parser import (
    InvescoHoldingsParseError,
    parse_official_invesco_holdings_payload,
)


def _payload(weights: list[float]) -> bytes:
    holdings = []
    for i, weight in enumerate(weights, start=1):
        holdings.append(
            {
                "ticker": f"T{i}",
                "issuerName": f"Issuer {i}",
                "percentageOfTotalNetAssets": weight,
                "cusip": f"{i:09d}",
            }
        )
    return json.dumps({"effectiveDate": "2026-04-24", "holdings": holdings}).encode("utf-8")


def test_official_json_with_holdings_parses_rows() -> None:
    parsed = parse_official_invesco_holdings_payload(
        raw_payload=_payload([60.0, 40.0]),
        source_url="https://dng-api.invesco.com/x",
        content_type="application/json",
    )
    assert len(parsed.frame.index) == 2
    assert parsed.parser_diagnostics["row_count"] == 2


def test_html_shell_is_rejected() -> None:
    with pytest.raises(InvescoHoldingsParseError, match="official_source_incomplete"):
        parse_official_invesco_holdings_payload(
            raw_payload=b"<!DOCTYPE html><html><head></head><body>shell</body></html>",
            source_url="https://www.invesco.com/us/financial-products/etfs/holdings",
            content_type="text/html",
        )


def test_empty_holdings_is_rejected() -> None:
    with pytest.raises(InvescoHoldingsParseError, match="empty_holdings"):
        parse_official_invesco_holdings_payload(
            raw_payload=json.dumps({"holdings": []}).encode("utf-8"),
            source_url="https://dng-api.invesco.com/x",
            content_type="application/json",
        )


def test_percentage_weights_converted_to_decimal() -> None:
    parsed = parse_official_invesco_holdings_payload(
        raw_payload=_payload([60.0, 40.0]),
        source_url="https://dng-api.invesco.com/x",
        content_type="application/json",
    )
    assert parsed.parser_diagnostics["weight_unit"] == "percentage"
    assert parsed.frame["normalized_weight"].sum() == pytest.approx(1.0)


def test_decimal_weights_remain_decimal() -> None:
    parsed = parse_official_invesco_holdings_payload(
        raw_payload=_payload([0.6, 0.4]),
        source_url="https://dng-api.invesco.com/x",
        content_type="application/json",
    )
    assert parsed.parser_diagnostics["weight_unit"] == "decimal"
    assert parsed.frame["normalized_weight"].sum() == pytest.approx(1.0)


def test_unknown_weight_unit_fails_closed() -> None:
    with pytest.raises(InvescoHoldingsParseError, match="weight_unit_unresolved"):
        parse_official_invesco_holdings_payload(
            raw_payload=_payload([4.2, 3.7]),
            source_url="https://dng-api.invesco.com/x",
            content_type="application/json",
        )
