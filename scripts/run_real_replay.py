from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qqq_cycle.backtest.real_replay import RealReplayConfig, run_real_replay


def main() -> None:
    # Use today's date as the default end to capture latest FRED/GPR data.
    today = datetime.now().strftime("%Y-%m-%d")
    
    # QQQ macro price source is required for full state calculations.
    # Note: run_real_replay will use this to populate the QQQ column in staging.
    # If the file doesn't exist yet, it will be missing in the first run.
    qqq_macro_path = ROOT / "cache" / "real_replay" / "raw" / "qqq_macro_captured.csv"
    
    result = run_real_replay(
        RealReplayConfig(
            cache_root=ROOT / "cache" / "real_replay",
            output_dir=ROOT / "outputs" / "replay" / "real",
            end=today,
            qqq_price_csv=qqq_macro_path if qqq_macro_path.exists() else None,
        )
    )
    print(f"mode={result.mode}")
    print(f"manifest={result.manifest_path}")
    print(f"weekly_replay={result.weekly_replay_path}")


if __name__ == "__main__":
    main()
