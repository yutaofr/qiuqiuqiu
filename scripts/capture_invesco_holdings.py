#!/usr/bin/env python3
"""Capture official Invesco QQQ holdings bytes and write machine proof."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def infer_extension(source_url: str, content_type: str | None) -> str:
    lowered = (content_type or "").lower()
    if "json" in lowered:
        return "json"
    if "spreadsheet" in lowered or "excel" in lowered:
        return "xlsx"
    if "csv" in lowered or "text/plain" in lowered:
        return "csv"
    suffix = Path(source_url.split("?", 1)[0]).suffix.lower().lstrip(".")
    return suffix if suffix in {"csv", "xlsx", "json"} else "csv"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week-end", default="2026-04-24")
    parser.add_argument("--asset", default="QQQ")
    parser.add_argument("--source-url", default=DEFAULT_INVESCO_QQQ_HOLDINGS_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("captures"))
    parser.add_argument("--sla-cutoff-utc", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    args = parser.parse_args()

    if args.asset.upper() != "QQQ":
        raise SystemExit("only QQQ is supported by the controlled backfill capture")
    if not source_url_allowed(args.source_url):
        raise SystemExit("source URL is not an allowed Invesco official source")

    raw, status, headers = fetch_bytes(args.source_url, args.timeout_seconds)
    fetched_at = utc_now_iso()
    extension = infer_extension(args.source_url, headers.get("content-type"))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"invesco_qqq_holdings_{args.week_end}_raw.{extension}"
    raw_path.write_bytes(raw)

    proof = PublicationProof(
        source_label="Invesco official QQQ holdings export",
        source_url=args.source_url,
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
    print(f"raw_capture={raw_path}")
    print(f"proof={proof_path}")
    print(f"strict_eligible={evaluated.strict_eligible}")
    print(f"strict_eligibility_reason={evaluated.strict_eligibility_reason}")


if __name__ == "__main__":
    main()
