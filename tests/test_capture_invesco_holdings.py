from __future__ import annotations

import hashlib
import io

import pandas as pd

from qqq_cycle.data_contracts.publication_proof import (
    PublicationProof,
    evaluate_publication_proof,
    sha256_bytes,
)
from scripts.capture_invesco_holdings import classify_payload


def test_http_200_html_shell_is_not_accepted_as_holdings() -> None:
    html = b"<!DOCTYPE html><html><head><title>QQQ</title></head><body>shell</body></html>"

    classification = classify_payload(
        html,
        "text/html; charset=UTF-8",
        "https://www.invesco.com/us/financial-products/etfs/holdings?audienceType=Investor&ticker=QQQ",
    )

    assert classification.capture_success is False
    assert classification.capture_reason == "official_source_incomplete"
    assert classification.payload_format == "html_shell"


def test_official_csv_fixture_parses_as_holdings() -> None:
    raw = (
        "ticker,issuerName,percentageOfTotalNetAssets\n"
        "NVDA,NVIDIA Corp,8.87\n"
        "AAPL,Apple Inc,7.55\n"
    ).encode("utf-8")

    classification = classify_payload(
        raw,
        "text/csv",
        "https://dng-api.invesco.com/cache/v1/accounts/en_US/shareclasses/QQQ/holdings/fund.csv",
    )

    assert classification.capture_success is True
    assert classification.payload_format == "csv"
    assert classification.holdings_rows == 2


def test_official_xlsx_fixture_parses_as_holdings() -> None:
    frame = pd.DataFrame(
        [
            {"ticker": "NVDA", "issuerName": "NVIDIA Corp", "percentageOfTotalNetAssets": 8.87},
            {"ticker": "AAPL", "issuerName": "Apple Inc", "percentageOfTotalNetAssets": 7.55},
        ]
    )
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    raw = buffer.getvalue()

    classification = classify_payload(
        raw,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "https://www.invesco.com/qqq.xlsx",
    )

    assert classification.capture_success is True
    assert classification.payload_format in {"xlsx", "xls"}
    assert classification.holdings_rows == 2


def test_content_sha256_is_computed_over_raw_bytes() -> None:
    raw = b"ticker,issuerName\nNVDA,NVIDIA Corp\n"
    expected = hashlib.sha256(raw).hexdigest()

    assert sha256_bytes(raw) == expected


def test_strict_eligible_remains_false_when_evidence_after_sla_cutoff() -> None:
    raw = b"ticker,issuerName\nNVDA,NVIDIA Corp\n"
    proof = PublicationProof(
        source_label="Invesco official QQQ holdings export",
        source_url="https://dng-api.invesco.com/cache/v1/accounts/en_US/shareclasses/QQQ/holdings/fund",
        content_sha256=sha256_bytes(raw),
        fetched_at_utc="2026-04-25T16:01:00Z",
        evidence_class="direct_http_capture_at_or_before_sla",
        evidence_timestamp_utc="2026-04-25T16:01:00Z",
        http_status=200,
        http_date_header=None,
        etag=None,
        last_modified_header=None,
        object_version_id=None,
        audit_log_sha256=None,
        third_party_snapshot_url=None,
        strict_eligible=True,
        strict_eligibility_reason="verified_direct_http_capture_before_sla",
    )

    evaluated = evaluate_publication_proof(
        proof,
        "2026-04-25T16:00:00Z",
        raw_payload=raw,
    )

    assert evaluated.strict_eligible is False
    assert evaluated.strict_eligibility_reason == "evidence_after_sla_cutoff"
