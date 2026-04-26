from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qqq_cycle.backtest.diagnostics import (
    build_replay_bundle,
    synthetic_replay_inputs,
    write_replay_outputs,
)
from qqq_cycle.backtest.oos_eval import (
    summarize_numerical_health,
    write_health_summary,
    write_tail_diagnostics,
)


def main() -> None:
    output_dir = ROOT / "outputs" / "replay" / "synthetic"
    bundle = build_replay_bundle(synthetic_replay_inputs())
    write_replay_outputs(bundle, output_dir)
    summary = summarize_numerical_health(bundle.weekly)
    write_health_summary(summary, output_dir)
    write_tail_diagnostics(bundle.weekly, output_dir)
    print(f"wrote replay outputs to {output_dir}")
    print(f"weekly_rows={len(bundle.weekly)}")
    print(f"drift_flag_count={summary['counts']['drift_flag_count']}")


if __name__ == "__main__":
    main()
