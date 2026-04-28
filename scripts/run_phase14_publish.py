"""Publish immutable Phase 14 snapshot artifacts from the latest live run.

Usage:
    python scripts/run_phase14_publish.py
    python scripts/run_phase14_publish.py --summary-path /path/to/live_run_summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qqq_cycle.ops.publishing import (  # noqa: E402
    DEFAULT_OPERATIONAL_SLA_CUTOFF,
    PublishingInputError,
    publish_from_live_summary_path,
)

DEFAULT_SUMMARY_PATH = Path("outputs/live/live_run_summary.json")
DEFAULT_OUTPUT_DIR = Path("outputs/phase14")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-path",
        default=str(DEFAULT_SUMMARY_PATH),
        help="Path to the live run summary JSON produced by scripts/run_live_pipeline.py",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Phase 14 output directory",
    )
    parser.add_argument(
        "--operational-sla-cutoff",
        default=DEFAULT_OPERATIONAL_SLA_CUTOFF,
        help="Operational SLA cutoff label embedded in published artifacts",
    )
    args = parser.parse_args(argv)

    try:
        snapshot, artifacts = publish_from_live_summary_path(
            summary_path=Path(args.summary_path),
            output_dir=Path(args.output_dir),
            operational_sla_cutoff=args.operational_sla_cutoff,
        )
    except (PublishingInputError, FileExistsError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("=== Phase 14 Publish ===")
    print(f"  week_end:              {snapshot.week_end}")
    print(f"  published_at:          {snapshot.published_at}")
    print(f"  mode:                  {snapshot.mode}")
    print(f"  source_hash:           {snapshot.source_hash}")
    print(f"  snapshot_history:      {artifacts.snapshot_history_path}")
    print(f"  snapshot_latest:       {artifacts.snapshot_latest_path}")
    print(f"  report_history:        {artifacts.report_history_path}")
    print(f"  report_latest:         {artifacts.report_latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
