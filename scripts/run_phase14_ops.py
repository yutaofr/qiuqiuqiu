"""Run Phase 14 operational alerts and dynamic ops status summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qqq_cycle.config import load_config  # noqa: E402
from qqq_cycle.ops.alerts import build_alert_log, write_alert_log  # noqa: E402
from qqq_cycle.ops.regime_monitor import load_latest_snapshot_per_week  # noqa: E402
from qqq_cycle.ops.revision_audit import build_revision_detail, load_snapshot_history, RevisionAuditInputError  # noqa: E402
from qqq_cycle.ops.status import build_ops_status_summary, write_ops_status_outputs  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("outputs/phase14")
DEFAULT_RUNBOOK_PATH = Path("docs/OPS_RUNBOOK.md")


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
        help="Phase 14 output directory for ops artifacts",
    )
    parser.add_argument(
        "--runbook-path",
        default=str(DEFAULT_RUNBOOK_PATH),
        help="Static runbook path referenced by the dynamic ops summary",
    )
    parser.add_argument(
        "--controlled-backfill-result",
        default=None,
        help="Optional controlled backfill result JSON to expose in ops status",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config()
        history = load_snapshot_history(Path(args.history_dir))
        revision_detail = build_revision_detail(history)
        latest_view = load_latest_snapshot_per_week(Path(args.history_dir))
        alert_log = build_alert_log(
            latest_view=latest_view,
            revision_detail=revision_detail,
            config=config,
        )
        alert_artifacts = write_alert_log(alert_log, output_dir=Path(args.output_dir))
        controlled_backfill_result = None
        if args.controlled_backfill_result is not None:
            controlled_backfill_result = json.loads(
                Path(args.controlled_backfill_result).read_text(encoding="utf-8")
            )
        summary = build_ops_status_summary(
            latest_view=latest_view,
            revision_detail=revision_detail,
            alert_log=alert_log,
            controlled_backfill_result=controlled_backfill_result,
            config=config,
            runbook_path=Path(args.runbook_path),
        )
        status_artifacts = write_ops_status_outputs(summary, output_dir=Path(args.output_dir))
    except RevisionAuditInputError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("=== Phase 14 Ops ===")
    print(f"  current_status:         {summary['current_status']}")
    print(f"  required_week_end:      {summary['required_week_end']}")
    print(f"  current_mode:           {summary['current_mode']}")
    print(f"  alert_count:            {len(summary['current_alerts'])}")
    print(f"  alert_log:              {alert_artifacts.alert_log_path}")
    print(f"  summary_json:           {status_artifacts.summary_json_path}")
    print(f"  summary_markdown:       {status_artifacts.summary_markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
