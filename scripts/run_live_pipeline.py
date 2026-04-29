"""Weekly live pipeline: single-week incremental run with state persistence.

Usage:
    python scripts/run_live_pipeline.py --week-end 2025-01-17
    python scripts/run_live_pipeline.py          # auto-detects last Friday

Prerequisites:
    - Run scripts/bootstrap_live_state.py once to seed state/live_state_latest/
    - cache/real_replay/staging/weekly_inputs.csv must contain the target week
    - cache/micro/ stores must be current for strict execution

Outputs:
    state/live_state_latest/        — updated state (also saved as dated archive)
    outputs/live/live_run_log.csv   — appended run record
    outputs/live/live_run_summary.json — latest run summary (overwritten)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qqq_cycle.live.controlled_backfill_contracts import resolve_live_contracts_for_week
from qqq_cycle.live.runtime import LiveRunResult, LiveRuntime
from qqq_cycle.live.state_io import StateNotAvailableError
from qqq_cycle.ops.backfill_ingest import load_controlled_backfill_result

STATE_DIR = Path("state")
OUTPUT_DIR = Path("outputs/live")
REAL_STAGING_CSV = Path("cache/real_replay/staging/weekly_inputs.csv")
MICRO_CACHE_DIR = Path("cache/micro")
PRODUCTION_LEDGER_DIR = Path("outputs/production_ledgers")
STORE_ROOT = Path("stores")


def _last_friday() -> str:
    today = pd.Timestamp.today().normalize()
    days_since_friday = (today.weekday() - 4) % 7
    last_fri = today - pd.Timedelta(days=days_since_friday)
    return last_fri.strftime("%Y-%m-%d")


def _load_macro_row(week_end: str) -> pd.Series:
    if not REAL_STAGING_CSV.exists():
        raise FileNotFoundError(
            f"Staging CSV not found: {REAL_STAGING_CSV}\n"
            "Ensure the weekly macro data has been updated."
        )
    df = pd.read_csv(REAL_STAGING_CSV, index_col=0, parse_dates=True)
    ts = pd.Timestamp(week_end)
    if ts not in df.index:
        available = df.index[-3:].strftime("%Y-%m-%d").tolist()
        raise KeyError(
            f"week_end {week_end} not found in staging CSV. "
            f"Latest available: {available}"
        )
    return df.loc[ts]


def _controlled_backfill_mode_for_week(week_end: str, phase14_output_dir: Path) -> str | None:
    controlled_result = load_controlled_backfill_result(
        week_end=week_end,
        asset="QQQ",
        output_dir=phase14_output_dir,
    )
    if controlled_result is None:
        return None
    if controlled_result.get("week_end") != week_end:
        return None
    return str(controlled_result["backfill_mode"])


def _print_result(result: LiveRunResult) -> None:
    def _fmt_metric(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    print(f"\n{'='*60}")
    print(f"  week_end:          {result.asof_week_end}")
    print(f"  pipeline mode:     {result.mode}")
    print(f"  execution_state:   {result.execution_state}")
    print(f"  execution_permitted: {result.execution_permitted}")
    if result.execution_block_reason:
        print(f"  block_reason:      {result.execution_block_reason}")
    if result.degraded_reason:
        print(f"  degraded_reason:   {result.degraded_reason}")
    s = result.signal_bundle
    print(
        f"  k_hat_t={s['k_hat_t']}  s_t={_fmt_metric(s['s_t'])}  "
        f"h_t={_fmt_metric(s['h_t'])}  "
        f"rho_t={_fmt_metric(s['rho_t'])}"
    )
    p = result.portfolio_bundle
    print(f"  omega_qqq_final={p['omega_qqq_final']:.4f}  "
          f"cb={p['circuit_breaker_active']}  "
          f"rebalance={p['rebalance_required']}")
    stale = [r["source_label"] for r in result.freshness_snapshot if not r["fresh_enough"]]
    if stale:
        print(f"  stale sources:     {stale}")
    print(f"  state saved to:    {result.state_path}")
    print(f"  run log:           {OUTPUT_DIR / 'live_run_log.csv'}")
    print("="*60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--week-end",
        default=None,
        help="ISO date of the Friday decision date (default: last Friday)",
    )
    parser.add_argument("--state-dir", default=str(STATE_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--phase14-output-dir",
        default="outputs/phase14",
        help="Directory containing controlled backfill result artifacts",
    )
    parser.add_argument(
        "--store-root",
        default=str(STORE_ROOT),
        help="Strict/backfill store root for controlled backfill weeks",
    )
    args = parser.parse_args()

    week_end = args.week_end or _last_friday()
    state_dir = Path(args.state_dir)
    output_dir = Path(args.output_dir)

    print(f"=== Live Pipeline Run: {week_end} ===")

    try:
        macro_row = _load_macro_row(week_end)
    except (FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    controlled_result = load_controlled_backfill_result(
        week_end=week_end,
        asset="QQQ",
        output_dir=Path(args.phase14_output_dir),
    )
    resolution = resolve_live_contracts_for_week(
        week_end=week_end,
        asset="QQQ",
        controlled_backfill_result=controlled_result,
        store_root=Path(args.store_root),
        legacy_cache_micro_dir=MICRO_CACHE_DIR,
        production_ledger_dir=PRODUCTION_LEDGER_DIR,
    )
    contracts = resolution.contracts
    backfill_mode = resolution.backfill_mode
    print(f"  contract_source:    {resolution.contract_source}")
    print(f"  strict_gate_passed: {resolution.strict_gate_passed}")
    print(f"  contract_reason:    {resolution.reason}")

    runtime = LiveRuntime()
    try:
        result = runtime.run_week(
            week_end=week_end,
            macro_row=macro_row,
            contracts=contracts,
            state_dir=state_dir,
            output_dir=output_dir,
            backfill_mode=backfill_mode,
            contract_source=resolution.contract_source,
            strict_gate_passed=resolution.strict_gate_passed,
        )
    except StateNotAvailableError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    _print_result(result)


if __name__ == "__main__":
    main()
