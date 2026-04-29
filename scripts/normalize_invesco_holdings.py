#!/usr/bin/env python3
"""Normalize captured Invesco QQQ holdings into the canonical namespace."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qqq_cycle.data_contracts.instrument_namespace import (
    normalize_holdings_namespace,
    write_normalized_holdings,
)


def read_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week-end", default="2026-04-24")
    parser.add_argument("--raw-path", type=Path, default=None)
    parser.add_argument("--canonical-master", type=Path, required=True)
    parser.add_argument("--share-class-map", type=Path, default=None)
    parser.add_argument("--override-ledger", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("normalized"))
    args = parser.parse_args()

    raw_path = args.raw_path or Path("captures") / f"invesco_qqq_holdings_{args.week_end}_raw.csv"
    raw = pd.read_csv(raw_path)
    master = pd.read_csv(args.canonical_master)
    normalized, summary = normalize_holdings_namespace(
        raw,
        master,
        share_class_map=read_optional_csv(args.share_class_map),
        override_ledger=read_optional_csv(args.override_ledger),
    )
    output_path = args.output_dir / f"qqq_holdings_{args.week_end}_normalized.csv"
    write_normalized_holdings(output_path, normalized)
    print(f"normalized={output_path}")
    print(f"weight_sum={summary.weight_sum:.12f}")
    print(f"unresolved_weight_sum={summary.unresolved_weight_sum:.12f}")
    print(f"unresolved_weight_blocks={summary.unresolved_weight_blocks}")
    print(f"namespace_version_hash={summary.namespace_version_hash}")


if __name__ == "__main__":
    main()
