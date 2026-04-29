#!/usr/bin/env python3
"""Run controlled QQQ backfill decision from captured proof and normalized holdings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qqq_cycle.data_contracts.backfill_validation import validate_normalized_holdings
from qqq_cycle.data_contracts.publication_proof import (
    PublicationProof,
    evaluate_publication_proof,
    read_publication_proof,
)
from qqq_cycle.ops.backfill_ingest import (
    controlled_backfill_result_from_decision,
    decide_backfill_scheme,
    write_backfill_stores,
    write_controlled_backfill_result,
)
from qqq_cycle.ops.revision_audit import append_controlled_backfill_revision_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week-end", default="2026-04-24")
    parser.add_argument("--asset", default="QQQ")
    parser.add_argument("--proof-path", type=Path, default=None)
    parser.add_argument("--raw-path", type=Path, default=None)
    parser.add_argument("--normalized-path", type=Path, default=None)
    parser.add_argument("--price-namespace-path", type=Path, default=None)
    parser.add_argument("--sla-cutoff-utc", default=None)
    parser.add_argument("--store-root", type=Path, default=Path("stores"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase14"))
    args = parser.parse_args()

    proof_path = args.proof_path or Path("captures") / f"invesco_qqq_holdings_{args.week_end}_proof.json"
    raw_path = args.raw_path or Path("captures") / f"invesco_qqq_holdings_{args.week_end}_raw.csv"
    normalized_path = args.normalized_path or Path("normalized") / f"qqq_holdings_{args.week_end}_normalized.csv"

    raw_payload: bytes | None = None
    if proof_path.exists() and raw_path.exists():
        proof = read_publication_proof(proof_path)
        raw_payload = raw_path.read_bytes()
        proof_result = evaluate_publication_proof(proof, args.sla_cutoff_utc, raw_payload=raw_payload)
    else:
        proof_result = PublicationProof(
            source_label="Invesco official QQQ holdings export",
            source_url="",
            content_sha256="",
            fetched_at_utc="",
            evidence_class="missing_machine_evidence",
            evidence_timestamp_utc=None,
            http_status=None,
            http_date_header=None,
            etag=None,
            last_modified_header=None,
            object_version_id=None,
            audit_log_sha256=None,
            third_party_snapshot_url=None,
            strict_eligible=False,
            strict_eligibility_reason="missing_machine_evidence",
        )
    if normalized_path.exists():
        normalized = pd.read_csv(normalized_path)
    else:
        normalized = pd.DataFrame(columns=["instrument_id", "normalized_weight", "normalization_status"])

    if args.price_namespace_path is not None and args.price_namespace_path.exists():
        price_namespace = pd.read_csv(args.price_namespace_path)
    else:
        price_namespace = None
    validation = validate_normalized_holdings(normalized, price_namespace)
    decision = decide_backfill_scheme(proof_result, validation)
    write_backfill_stores(
        normalized_holdings=normalized,
        decision=decision,
        week_end=args.week_end,
        asset=args.asset,
        store_root=args.store_root,
    )
    result = controlled_backfill_result_from_decision(
        week_end=args.week_end,
        asset=args.asset,
        proof_result=proof_result,
        validation_result=validation,
        decision=decision,
        normalized_holdings=normalized,
    )
    result_path = write_controlled_backfill_result(result, output_dir=args.output_dir)
    revision_path = append_controlled_backfill_revision_record(result, output_dir=args.output_dir)
    print(f"scheme={decision.scheme}")
    print(f"decision_reason={decision.reason}")
    print(f"validation_reason={validation.validation_reason}")
    print(f"strict_eligible={proof_result.strict_eligible}")
    print(f"result={result_path}")
    print(f"revision_record={revision_path}")


if __name__ == "__main__":
    main()
