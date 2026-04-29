#!/usr/bin/env python3
"""Normalize captured Invesco QQQ holdings into the canonical namespace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qqq_cycle.data_contracts.instrument_namespace import (
    NamespaceNormalizationError,
    normalize_holdings_namespace,
    write_normalized_holdings,
)
from qqq_cycle.data_contracts.invesco_holdings_parser import (
    InvescoHoldingsParseError,
    parse_official_invesco_holdings_payload,
)


def read_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return pd.read_csv(path)


def load_raw_holdings(raw_path: Path) -> tuple[pd.DataFrame, dict]:
    suffix = raw_path.suffix.lower()
    if suffix == ".json":
        raw_payload = raw_path.read_bytes()
        parsed = parse_official_invesco_holdings_payload(
            raw_payload=raw_payload,
            source_url=str(raw_path),
            content_type="application/json",
        )
        return parsed.frame, parsed.parser_diagnostics

    frame = pd.read_csv(raw_path)
    diagnostics = {
        "source_url": str(raw_path),
        "content_type": "text/csv",
        "row_count": int(len(frame.index)),
        "raw_weight_sum": float(pd.to_numeric(frame.get("raw_weight", frame.get("weight")), errors="coerce").sum()),
        "normalized_weight_sum": None,
        "weight_unit": "unknown",
        "effective_date": None,
    }
    return frame, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week-end", default="2026-04-24")
    parser.add_argument("--raw-path", type=Path, default=None)
    parser.add_argument("--canonical-master", type=Path, required=True)
    parser.add_argument("--share-class-map", type=Path, default=None)
    parser.add_argument("--override-ledger", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("normalized"))
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    default_raw = Path("captures") / f"invesco_qqq_holdings_{args.week_end}_raw.json"
    raw_path = args.raw_path or (default_raw if default_raw.exists() else default_raw.with_suffix(".csv"))
    output_path = args.output_path or (args.output_dir / f"qqq_holdings_{args.week_end}_normalized.csv")

    try:
        raw, parser_diagnostics = load_raw_holdings(raw_path)
        master = pd.read_csv(args.canonical_master)
        normalized, summary = normalize_holdings_namespace(
            raw,
            master,
            share_class_map=read_optional_csv(args.share_class_map),
            override_ledger=read_optional_csv(args.override_ledger),
            asof_week_end=args.week_end,
        )
        write_normalized_holdings(output_path, normalized)
    except (InvescoHoldingsParseError, NamespaceNormalizationError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc

    print(f"normalized={output_path}")
    print(f"weight_sum={summary.weight_sum:.12f}")
    print(f"unresolved_weight_sum={summary.unresolved_weight_sum:.12f}")
    print(f"unresolved_weight_blocks={summary.unresolved_weight_blocks}")
    print(f"namespace_version_hash={summary.namespace_version_hash}")
    print(f"parser_diagnostics={json.dumps(parser_diagnostics, sort_keys=True)}")


if __name__ == "__main__":
    main()
