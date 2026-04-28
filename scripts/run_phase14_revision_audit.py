"""Run the Phase 14 immutable-history revision audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qqq_cycle.ops.revision_audit import (  # noqa: E402
    RevisionAuditInputError,
    build_revision_tests,
    build_revision_summary,
    build_revision_detail,
    load_snapshot_history,
    write_revision_audit_outputs,
)

DEFAULT_OUTPUT_DIR = Path("outputs/phase14")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history-dir",
        default=str(DEFAULT_OUTPUT_DIR / "history"),
        help="Immutable Phase 14 snapshot history directory",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Phase 14 output directory for revision artifacts",
    )
    args = parser.parse_args(argv)

    try:
        history = load_snapshot_history(Path(args.history_dir))
        detail = build_revision_detail(history)
        summary = build_revision_summary(detail)
        tests_payload = build_revision_tests(detail)
        artifacts = write_revision_audit_outputs(
            history_dir=Path(args.history_dir),
            output_dir=Path(args.output_dir),
        )
    except RevisionAuditInputError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("=== Phase 14 Revision Audit ===")
    print(f"  weeks_total:             {int(summary.iloc[0]['weeks_total'])}")
    print(f"  multi_run_weeks:         {int(summary.iloc[0]['weeks_with_multiple_runs'])}")
    print(f"  material_revision_weeks: {int(summary.iloc[0]['material_revision_weeks'])}")
    print(f"  summary_csv:             {artifacts.summary_csv_path}")
    print(f"  detail_csv:              {artifacts.detail_csv_path}")
    print(f"  tests_json:              {artifacts.tests_json_path}")
    print(f"  checks:                  {json.dumps(tests_payload['checks'], ensure_ascii=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
