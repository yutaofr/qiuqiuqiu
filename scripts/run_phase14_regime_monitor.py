"""Run the Phase 14 latest-view regime monitor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qqq_cycle.ops.regime_monitor import (  # noqa: E402
    build_event_response_summary,
    build_state_duration_summary,
    build_state_transition_matrix,
    load_latest_snapshot_per_week,
    write_regime_monitor_outputs,
)
from qqq_cycle.ops.revision_audit import RevisionAuditInputError  # noqa: E402

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
        help="Phase 14 output directory for regime-monitor artifacts",
    )
    args = parser.parse_args(argv)

    try:
        latest_view = load_latest_snapshot_per_week(Path(args.history_dir))
        transition_matrix = build_state_transition_matrix(latest_view)
        duration_summary = build_state_duration_summary(latest_view)
        event_response_summary = build_event_response_summary(latest_view)
        artifacts = write_regime_monitor_outputs(
            history_dir=Path(args.history_dir),
            output_dir=Path(args.output_dir),
        )
    except RevisionAuditInputError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("=== Phase 14 Regime Monitor ===")
    print(f"  weeks_total:             {len(latest_view)}")
    print(f"  transition_pairs:        {int(transition_matrix.to_numpy().sum())}")
    print(f"  duration_rows:           {len(duration_summary)}")
    print(f"  event_response_rows:     {len(event_response_summary)}")
    print(f"  transition_matrix:       {artifacts.transition_matrix_path}")
    print(f"  duration_summary:        {artifacts.duration_summary_path}")
    print(f"  event_response_summary:  {artifacts.event_response_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
