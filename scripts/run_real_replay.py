from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qqq_cycle.backtest.real_replay import RealReplayConfig, run_real_replay


def main() -> None:
    result = run_real_replay(
        RealReplayConfig(
            cache_root=ROOT / "cache" / "real_replay",
            output_dir=ROOT / "outputs" / "replay" / "real",
        )
    )
    print(f"mode={result.mode}")
    print(f"manifest={result.manifest_path}")
    print(f"weekly_replay={result.weekly_replay_path}")


if __name__ == "__main__":
    main()
