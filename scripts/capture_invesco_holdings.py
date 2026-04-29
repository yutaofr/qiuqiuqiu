#!/usr/bin/env python3
"""Capture official Invesco QQQ holdings bytes and write machine proof."""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qqq_cycle.data_contracts.publication_proof import (
    PublicationProof,
    evaluate_publication_proof,
    sha256_bytes,
    source_url_allowed,
    write_publication_proof,
)


DEFAULT_INVESCO_QQQ_HOLDINGS_URL = (
    "https://www.invesco.com/us/financial-products/etfs/holdings?audienceType=Investor&ticker=QQQ"
)
DEFAULT_INVESCO_QQQ_HOLDINGS_JSON_URL = (
    "https://dng-api.invesco.com/cache/v1/accounts/en_US/shareclasses/QQQ/holdings/fund"
    "?idType=ticker&interval=monthly&productType=ETF&loadType=initial"
)
DEFAULT_INVESCO_QQQ_HOLDINGS_JSON_FALLBACK_URL = (
    "https://dng-api.invesco.com/cache/v1/accounts/en_US/shareclasses/QQQ/holdings/fund"
    "?idType=ticker&interval=monthly&productType=ETF"
)
DEFAULT_PROBE_URLS = (
    DEFAULT_INVESCO_QQQ_HOLDINGS_URL,
    DEFAULT_INVESCO_QQQ_HOLDINGS_JSON_FALLBACK_URL,
    DEFAULT_INVESCO_QQQ_HOLDINGS_JSON_URL,
)

REQUIRED_HOLDINGS_COLUMNS = ("raw_symbol", "ticker", "issuerName", "issuer_name")


@dataclass(frozen=True)
class PayloadClassification:
    payload_format: str
    capture_success: bool
    capture_reason: str
    extension: str
    holdings_rows: int


@dataclass(frozen=True)
class ProbeOutcome:
    source_url: str
    status: int | None
    headers: dict[str, str]
    raw: bytes
    classification: PayloadClassification


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def infer_extension(source_url: str, content_type: str | None) -> str:
    lowered = (content_type or "").lower()
    if "html" in lowered:
        return "html"
    if "json" in lowered:
        return "json"
    if "spreadsheet" in lowered or "excel" in lowered:
        return "xlsx"
    if "ms-excel" in lowered:
        return "xls"
    if "csv" in lowered or "text/plain" in lowered:
        return "csv"
    suffix = Path(source_url.split("?", 1)[0]).suffix.lower().lstrip(".")
    return suffix if suffix in {"csv", "xlsx", "xls", "json", "html"} else "bin"


def fetch_bytes(source_url: str, timeout_seconds: int) -> tuple[bytes, int | None, dict[str, str]]:
    request = Request(source_url, headers={"User-Agent": "qqq-cycle-controlled-backfill/1.0"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            headers = {key.lower(): value for key, value in response.headers.items()}
            return response.read(), int(response.status), headers
    except HTTPError as exc:
        return exc.read(), int(exc.code), {key.lower(): value for key, value in exc.headers.items()}
    except URLError as exc:
        raise RuntimeError(f"failed to capture Invesco holdings: {exc}") from exc


def _looks_like_html(raw: bytes, content_type: str | None) -> bool:
    if "html" in (content_type or "").lower():
        return True
    prefix = raw[:2048].decode("utf-8", errors="ignore").lower()
    return "<!doctype html" in prefix or "<html" in prefix or "<head" in prefix


def _extract_csv_rows(raw: bytes) -> int:
    frame = pd.read_csv(io.BytesIO(raw))
    if frame.empty:
        return 0
    if not any(col in frame.columns for col in REQUIRED_HOLDINGS_COLUMNS):
        return 0
    return int(len(frame.index))


def _extract_excel_rows(raw: bytes) -> int:
    frame = pd.read_excel(io.BytesIO(raw))
    if frame.empty:
        return 0
    if not any(col in frame.columns for col in REQUIRED_HOLDINGS_COLUMNS):
        return 0
    return int(len(frame.index))


def _extract_json_rows(raw: bytes) -> int:
    payload = json.loads(raw.decode("utf-8"))
    if isinstance(payload, dict):
        holdings = payload.get("holdings")
        if isinstance(holdings, list):
            return len(holdings)
    if isinstance(payload, list):
        return len(payload)
    return 0


def classify_payload(raw: bytes, content_type: str | None, source_url: str) -> PayloadClassification:
    if _looks_like_html(raw, content_type):
        return PayloadClassification(
            payload_format="html_shell",
            capture_success=False,
            capture_reason="official_source_incomplete",
            extension="html",
            holdings_rows=0,
        )

    extension = infer_extension(source_url, content_type)
    parser_errors: list[str] = []
    detectors: list[tuple[str, str, Callable[[bytes], int]]] = [
        ("csv", "capture_success_machine_readable_csv", _extract_csv_rows),
        ("xls", "capture_success_machine_readable_xls", _extract_excel_rows),
        ("xlsx", "capture_success_machine_readable_xlsx", _extract_excel_rows),
        ("json", "capture_success_machine_readable_json", _extract_json_rows),
    ]
    # Prefer extension-implied parser first, then the rest.
    detectors.sort(key=lambda item: 0 if item[0] == extension else 1)
    for payload_format, reason, parser in detectors:
        try:
            rows = int(parser(raw))
        except Exception as exc:  # pragma: no cover - diagnostics path
            parser_errors.append(f"{payload_format}:{type(exc).__name__}")
            continue
        if rows > 0:
            return PayloadClassification(
                payload_format=payload_format,
                capture_success=True,
                capture_reason=reason,
                extension=payload_format,
                holdings_rows=rows,
            )

    capture_reason = "official_source_incomplete" if parser_errors else "unsupported_payload_type"
    return PayloadClassification(
        payload_format=extension if extension != "bin" else "unknown",
        capture_success=False,
        capture_reason=capture_reason,
        extension=extension,
        holdings_rows=0,
    )


def probe_official_endpoints(
    source_urls: list[str],
    timeout_seconds: int,
) -> list[ProbeOutcome]:
    outcomes: list[ProbeOutcome] = []
    for source_url in source_urls:
        if not source_url_allowed(source_url):
            continue
        raw, status, headers = fetch_bytes(source_url, timeout_seconds)
        classification = classify_payload(raw, headers.get("content-type"), source_url)
        outcomes.append(
            ProbeOutcome(
                source_url=source_url,
                status=status,
                headers=headers,
                raw=raw,
                classification=classification,
            )
        )
    return outcomes


def choose_probe_result(outcomes: list[ProbeOutcome]) -> ProbeOutcome:
    for outcome in outcomes:
        if outcome.status == 200 and outcome.classification.capture_success:
            return outcome
    if outcomes:
        return outcomes[0]
    raise RuntimeError("no allowed official Invesco source URL was probeable")


def write_capture_status(
    *,
    output_dir: Path,
    week_end: str,
    asset: str,
    selected: ProbeOutcome,
    outcomes: list[ProbeOutcome],
    raw_path: Path,
    proof_path: Path,
) -> Path:
    payload = {
        "week_end": week_end,
        "asset": asset,
        "capture_success": selected.classification.capture_success,
        "capture_reason": selected.classification.capture_reason,
        "source_url": selected.source_url,
        "http_status": selected.status,
        "content_type": selected.headers.get("content-type"),
        "payload_format": selected.classification.payload_format,
        "holdings_rows": selected.classification.holdings_rows,
        "raw_path": str(raw_path),
        "proof_path": str(proof_path),
        "probed_sources": [
            {
                "source_url": outcome.source_url,
                "http_status": outcome.status,
                "content_type": outcome.headers.get("content-type"),
                **asdict(outcome.classification),
            }
            for outcome in outcomes
        ],
    }
    status_path = output_dir / f"invesco_qqq_holdings_{week_end}_capture_status.json"
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week-end", default="2026-04-24")
    parser.add_argument("--asset", default="QQQ")
    parser.add_argument("--source-url", default=DEFAULT_INVESCO_QQQ_HOLDINGS_URL)
    parser.add_argument(
        "--probe-official-endpoints",
        action="store_true",
        help="Probe supported official Invesco endpoints and select first machine-readable holdings payload",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("captures"))
    parser.add_argument("--sla-cutoff-utc", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    args = parser.parse_args()

    if args.asset.upper() != "QQQ":
        raise SystemExit("only QQQ is supported by the controlled backfill capture")
    source_urls = (
        list(DEFAULT_PROBE_URLS)
        if args.probe_official_endpoints
        else [args.source_url]
    )
    for source_url in source_urls:
        if not source_url_allowed(source_url):
            raise SystemExit("source URL is not an allowed Invesco official source")

    outcomes = probe_official_endpoints(source_urls, args.timeout_seconds)
    selected = choose_probe_result(outcomes)
    raw = selected.raw
    status = selected.status
    headers = selected.headers
    classification = selected.classification
    fetched_at = utc_now_iso()
    extension = classification.extension
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"invesco_qqq_holdings_{args.week_end}_raw.{extension}"
    raw_path.write_bytes(raw)

    proof = PublicationProof(
        source_label="Invesco official QQQ holdings export",
        source_url=selected.source_url,
        content_sha256=sha256_bytes(raw),
        fetched_at_utc=fetched_at,
        evidence_class="direct_http_capture_at_or_before_sla",
        evidence_timestamp_utc=fetched_at,
        http_status=status,
        http_date_header=headers.get("date"),
        etag=headers.get("etag"),
        last_modified_header=headers.get("last-modified"),
        object_version_id=None,
        audit_log_sha256=None,
        third_party_snapshot_url=None,
        strict_eligible=False,
        strict_eligibility_reason="missing_sla_cutoff",
    )
    evaluated = evaluate_publication_proof(proof, args.sla_cutoff_utc, raw_payload=raw)
    proof_path = output_dir / f"invesco_qqq_holdings_{args.week_end}_proof.json"
    write_publication_proof(proof_path, evaluated)
    status_path = write_capture_status(
        output_dir=output_dir,
        week_end=args.week_end,
        asset=args.asset.upper(),
        selected=selected,
        outcomes=outcomes,
        raw_path=raw_path,
        proof_path=proof_path,
    )
    print(f"raw_capture={raw_path}")
    print(f"proof={proof_path}")
    print(f"capture_status={status_path}")
    print(f"capture_success={classification.capture_success}")
    print(f"capture_reason={classification.capture_reason}")
    print(f"source_url={selected.source_url}")
    print(f"payload_format={classification.payload_format}")
    print(f"holdings_rows={classification.holdings_rows}")
    print(f"strict_eligible={evaluated.strict_eligible}")
    print(f"strict_eligibility_reason={evaluated.strict_eligibility_reason}")


if __name__ == "__main__":
    main()
