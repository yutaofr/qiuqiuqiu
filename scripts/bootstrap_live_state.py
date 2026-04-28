"""One-time bootstrap: build live_state_latest/ from a full historical replay.

Usage:
    python scripts/bootstrap_live_state.py [--state-dir state] [--output-dir outputs/live]

This script replays the full pipeline history to produce the final covariance,
prototype, IIR envelope, and circuit-breaker state, then serializes it to
state/live_state_latest/.  Run once before starting weekly live runs.

Re-running overwrites the existing state — use the dated archive copies to
recover a prior state if needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qqq_cycle.config import load_config
from qqq_cycle.core.covariance import RobustEWCov2D
from qqq_cycle.core.drift_probe import DriftProbe
from qqq_cycle.core.proto_online import initialize_prototypes_from_history
from qqq_cycle.core.state_layer import compute_state_layer
from qqq_cycle.core.stress_layer import compute_stress_layer
from qqq_cycle.live.runtime import _run_pipeline_step, _run_portfolio_step
from qqq_cycle.live.state_io import LiveState, save_state
from qqq_cycle.pipeline import (
    PipelineContracts,
    _check_strict_gate,
    _safe_float,
)
from qqq_cycle.portfolio.construction import BacktestConfig

REAL_STAGING_CSV = Path("cache/real_replay/staging/weekly_inputs.csv")
MICRO_CACHE_DIR = Path("cache/micro")
PHASE11_WEIGHTS_CSV = Path("outputs/phase11/weekly_weights.csv")


def _load_real_macro(staging_csv: Path) -> pd.DataFrame:
    if not staging_csv.exists():
        raise FileNotFoundError(
            f"Staging CSV not found: {staging_csv}\n"
            "Run scripts/run_pipeline.py --mode strict_real first."
        )
    df = pd.read_csv(staging_csv, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _load_micro_contracts() -> PipelineContracts | None:
    """Try to load real micro stores; return None if unavailable."""
    try:
        from qqq_cycle.data_contracts.constituents import CsvConstituentStore
        from qqq_cycle.data_contracts.corp_actions import InMemoryCorporateActionStore
        from qqq_cycle.data_contracts.pit_adjustment import LedgerPITAdjustmentEngine
        from qqq_cycle.data_contracts.raw_prices import CsvRawPriceStore
        from qqq_cycle.data_contracts.symbol_identity import InMemorySymbolIdentityResolver
        from qqq_cycle.data_contracts.weights import CsvWeightStore

        prices_dir = MICRO_CACHE_DIR / "prices"
        rows: list[pd.DataFrame] = []
        for path in sorted(prices_dir.glob("*.csv")):
            df = pd.read_csv(path, usecols=["trade_date", "raw_close", "asof_timestamp"])
            df["ticker"] = path.stem.upper()
            df["source_label"] = "local_seed_raw_close"
            rows.append(df[["trade_date", "ticker", "raw_close", "source_label", "asof_timestamp"]])

        if not rows:
            return None

        ledger_path = Path("outputs/production_ledgers/raw_prices.csv")
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
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
            constituent_store=CsvConstituentStore(MICRO_CACHE_DIR / "constituents.csv"),
            weight_store=CsvWeightStore(MICRO_CACHE_DIR / "weights.csv"),
        )
    except Exception as exc:
        print(f"  [warn] micro stores unavailable ({exc}); bootstrapping in degraded mode")
        return None


def _load_prev_omega_qqq() -> float:
    """Read the last executed QQQ weight from Phase 11 output, or default to 0.5."""
    if PHASE11_WEIGHTS_CSV.exists():
        try:
            df = pd.read_csv(PHASE11_WEIGHTS_CSV)
            if "omega_qqq_final" in df.columns and len(df) > 0:
                return float(df["omega_qqq_final"].iloc[-1])
        except Exception:
            pass
    return 0.5


def run_bootstrap(state_dir: Path, output_dir: Path) -> None:
    print("=== Bootstrap Live State ===")
    config = load_config()
    macro_df = _load_real_macro(REAL_STAGING_CSV)
    print(f"  loaded macro: {len(macro_df)} weeks, {macro_df.index[0].date()} – {macro_df.index[-1].date()}")

    contracts = _load_micro_contracts()
    print(f"  contracts: {'strict' if contracts else 'degraded (no micro stores)'}")

    # Resolve contracts the same way run_pipeline does (precompute weekly_h_t from stores).
    from qqq_cycle.pipeline import _compute_weekly_h_t_from_stores
    if contracts is not None and contracts.pit_engine is not None:
        print("  computing weekly h_t from micro stores (this may take a few minutes) ...")
        computed_h_t = _compute_weekly_h_t_from_stores(macro_df.index, contracts, config)
        contracts = PipelineContracts(
            weekly_h_t=computed_h_t,
            pit_engine=contracts.pit_engine,
            constituent_store=contracts.constituent_store,
            weight_store=contracts.weight_store,
            pit_engine_available=True,
            constituents_available=True,
            weights_available=True,
        )

    can_compute_h_t, degraded_reason = _check_strict_gate(contracts)

    # Precompute batch layers on the full history.
    print("  computing batch layers ...")
    state_frame = compute_state_layer(macro_df)
    theta = state_frame[["H", "I"]]
    stress_result = compute_stress_layer(theta, state_frame["E"])
    stress_frame = stress_result.frame
    drift_frame = DriftProbe(
        theta_lo=config.drift.theta_lo,
        theta_hi=config.drift.theta_hi,
    ).compute(macro_df)

    finite_theta = theta.dropna()
    if len(finite_theta) < 20:
        raise RuntimeError("Insufficient finite theta rows to initialize covariance.")

    cov = RobustEWCov2D(warmup_weeks=config.warmup_weeks)
    cov_state = cov.initialize_from_history(finite_theta.iloc[:20].to_numpy())
    proto = None
    proto_seed: list[np.ndarray] = []
    h_t_lead_prev: float = 0.0
    heal_count: int = 0

    # Replay all weeks to build up state.
    print(f"  replaying {len(theta)} weeks ...")
    for week_idx, (week_end, theta_row_s) in enumerate(theta.iterrows()):
        theta_row_vals = theta_row_s.to_numpy(dtype=float)

        s_t_val = (
            _safe_float(stress_frame.at[week_end, "s"])
            if week_end in stress_frame.index else None
        )
        i_t_val = float(theta_row_vals[1]) if np.isfinite(theta_row_vals[1]) else None
        drift_raw = (
            _safe_float(drift_frame.at[week_end, "drift_probe_raw"])
            if week_end in drift_frame.index else None
        )
        drift_flag = (
            int(drift_frame.at[week_end, "drift_flag"])
            if week_end in drift_frame.index
            and pd.notna(drift_frame.at[week_end, "drift_flag"])
            else 0
        )

        h_t_raw_val: float | None = None
        if can_compute_h_t and contracts is not None and contracts.weekly_h_t is not None:
            if week_end in contracts.weekly_h_t.index:
                h_t_raw_val = _safe_float(contracts.weekly_h_t.loc[week_end])

        (
            _,
            cov_state,
            proto,
            proto_seed,
            h_t_lead_prev,
            heal_count,
        ) = _run_pipeline_step(
            week_end=week_end,
            theta_row=theta_row_vals,
            s_t_val=s_t_val,
            i_t_val=i_t_val,
            drift_raw=drift_raw,
            drift_flag=drift_flag,
            h_t_raw=h_t_raw_val,
            cov_state=cov_state,
            proto=proto,
            proto_seed=proto_seed,
            h_t_lead_prev=h_t_lead_prev,
            heal_count=heal_count,
            config=config,
            can_compute_h_t=can_compute_h_t,
            degraded_reason=degraded_reason,
            week_index=week_idx,
            state_frame=state_frame,
            stress_frame=stress_frame,
            drift_frame=drift_frame,
        )

    last_week_end = theta.index[-1].strftime("%Y-%m-%d")
    prev_omega_qqq = _load_prev_omega_qqq()
    print(f"  prev_omega_qqq from Phase 11: {prev_omega_qqq:.4f}")

    final_state = LiveState(
        week_end=last_week_end,
        cov_state=cov_state,
        proto=proto,
        proto_seed=proto_seed,
        h_t_lead_prev=h_t_lead_prev,
        heal_count=heal_count,
        warmup_count=cov_state.warmup_count,
        breaker_active=False,
        weeks_outside_s1=0,
        prev_omega_qqq=prev_omega_qqq,
        macro_tail=macro_df,
        last_successful_timestamps={"bootstrap": pd.Timestamp.now().isoformat()},
    )

    state_dir.mkdir(parents=True, exist_ok=True)
    save_state(final_state, state_dir)
    print(f"  saved live state → {state_dir / 'live_state_latest'}")
    print(f"  warmup_count={cov_state.warmup_count}, proto={'initialized' if proto else 'None'}")
    print(f"  last week_end: {last_week_end}")
    print("Bootstrap complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", default="state", help="State root directory")
    parser.add_argument("--output-dir", default="outputs/live", help="Live output directory")
    args = parser.parse_args()
    run_bootstrap(Path(args.state_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
