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

from qqq_cycle.data_contracts.constituents import CsvConstituentStore
from qqq_cycle.data_contracts.corp_actions import InMemoryCorporateActionStore
from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError, LedgerPITAdjustmentEngine
from qqq_cycle.data_contracts.raw_prices import CsvRawPriceStore
from qqq_cycle.data_contracts.symbol_identity import InMemorySymbolIdentityResolver
from qqq_cycle.data_contracts.weights import CsvWeightStore
from qqq_cycle.live.runtime import LiveRunResult, LiveRuntime
from qqq_cycle.live.state_io import StateNotAvailableError
from qqq_cycle.pipeline import PipelineContracts

STATE_DIR = Path("state")
OUTPUT_DIR = Path("outputs/live")
REAL_STAGING_CSV = Path("cache/real_replay/staging/weekly_inputs.csv")
MICRO_CACHE_DIR = Path("cache/micro")
PRODUCTION_LEDGER_DIR = Path("outputs/production_ledgers")


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


def _build_contracts() -> PipelineContracts | None:
    """Build micro contracts from cached stores, or return None."""
    const_csv = MICRO_CACHE_DIR / "constituents.csv"
    weights_csv = MICRO_CACHE_DIR / "weights.csv"
    prices_dir = MICRO_CACHE_DIR / "prices"

    if not const_csv.exists() or not weights_csv.exists():
        print("  [warn] micro stores not found — running in degraded mode")
        return None

    rows: list[pd.DataFrame] = []
    for path in sorted(prices_dir.glob("*.csv")):
        df = pd.read_csv(path, usecols=["trade_date", "raw_close", "asof_timestamp"])
        df["ticker"] = path.stem.upper()
        df["source_label"] = "local_seed_raw_close"
        rows.append(df[["trade_date", "ticker", "raw_close", "source_label", "asof_timestamp"]])

    if not rows:
        print("  [warn] no raw price files in cache/micro/prices/ — running in degraded mode")
        return None

    PRODUCTION_LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = PRODUCTION_LEDGER_DIR / "raw_prices.csv"
    pd.concat(rows, ignore_index=True).sort_values(["ticker", "trade_date"]).to_csv(
        ledger_path, index=False
    )

    pit_engine = LedgerPITAdjustmentEngine(
        raw_price_store=CsvRawPriceStore(ledger_path),
        corporate_action_store=InMemoryCorporateActionStore([]),
        identity_resolver=InMemorySymbolIdentityResolver([]),
    )
    return PipelineContracts(
        pit_engine=pit_engine,
        constituent_store=CsvConstituentStore(const_csv),
        weight_store=CsvWeightStore(weights_csv),
    )


def _print_result(result: LiveRunResult) -> None:
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
    print(f"  k_hat_t={s['k_hat_t']}  s_t={s['s_t']:.4f if s['s_t'] else 'n/a'}  "
          f"h_t={s['h_t']:.4f if s['h_t'] else 'n/a'}  "
          f"rho_t={s['rho_t']:.4f if s['rho_t'] else 'n/a'}")
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

    contracts = _build_contracts()

    runtime = LiveRuntime()
    try:
        result = runtime.run_week(
            week_end=week_end,
            macro_row=macro_row,
            contracts=contracts,
            state_dir=state_dir,
            output_dir=output_dir,
        )
    except StateNotAvailableError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    _print_result(result)


if __name__ == "__main__":
    main()
