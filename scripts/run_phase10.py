"""Phase 10: Regime separation and signal predictive power analysis.

Usage:
    python scripts/run_phase10.py [--force-refresh]

Reads:
    outputs/pipeline/strict_real_pipeline_output.csv
    outputs/pipeline/pipeline_mode_summary.json

Writes:
    outputs/phase10/qqq_weekly_aligned.csv
    outputs/phase10/regime_separation_summary.csv
    outputs/phase10/regime_separation_tests.json
    outputs/phase10/signal_predictive_summary.csv
    outputs/phase10/phase10_acceptance.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qqq_cycle.backtest.phase10_analysis import (
    HORIZONS,
    REALIZED_METRICS,
    SIGNALS,
    check_signal_predictive_pass,
    compute_forward_metrics,
    compute_realized_metrics,
    fetch_qqq_weekly,
    load_strict_rows,
    run_regime_separation,
    run_signal_predictive,
    HIGH_CONF_THRESHOLD,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_CSV = REPO_ROOT / "outputs/pipeline/strict_real_pipeline_output.csv"
PIPELINE_SUMMARY = REPO_ROOT / "outputs/pipeline/pipeline_mode_summary.json"
OUTPUT_DIR = REPO_ROOT / "outputs/phase10"
QQQ_CACHE = REPO_ROOT / "cache/phase10/qqq_weekly_price.csv"


def main(force_refresh: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load summary for epoch bounds ---
    with open(PIPELINE_SUMMARY) as f:
        summary = json.load(f)

    epoch_start = summary["production_strict_epoch_start"]
    strict_last = summary["strict_last_valid_week"]
    print(f"[phase10] epoch: {epoch_start} → {strict_last}")

    # --- Load strict rows ---
    print("[phase10] Loading strict rows...")
    df = load_strict_rows(PIPELINE_CSV)
    print(f"[phase10] {len(df)} strict rows loaded")

    # --- Fetch QQQ weekly prices ---
    print("[phase10] Fetching QQQ weekly prices...")
    qqq_weekly = fetch_qqq_weekly(
        start=epoch_start,
        end=(pd.Timestamp(strict_last) + pd.DateOffset(weeks=10)).strftime("%Y-%m-%d"),
        cache_path=QQQ_CACHE,
        force_refresh=force_refresh,
    )
    print(f"[phase10] QQQ weekly: {len(qqq_weekly)} rows")

    # --- Build daily-resolution price series for metric computation ---
    # We need daily prices to compute realized vol and forward metrics.
    # Reload from yfinance or use cached daily prices.
    qqq_daily = _load_qqq_daily(epoch_start, strict_last, force_refresh)

    # --- Compute realized metrics (backward-looking) ---
    print("[phase10] Computing realized metrics...")
    realized_df = compute_realized_metrics(qqq_daily, df["week_end"])
    realized_df = realized_df.reset_index()

    # --- Compute forward metrics ---
    print("[phase10] Computing forward metrics...")
    forward_df = compute_forward_metrics(qqq_daily, df["week_end"], HORIZONS)
    forward_df = forward_df.reset_index()

    # --- Merge everything ---
    merged = df.merge(realized_df, on="week_end", how="left")
    merged = merged.merge(forward_df, on="week_end", how="left")

    # Also attach qqq_adj_close from weekly for reference
    qqq_ref = qqq_weekly.rename(columns={"qqq_adj_close": "qqq_adj_close_weekly"})
    # Normalize datetime precision to avoid merge_asof dtype mismatch
    merged["week_end"] = merged["week_end"].astype("datetime64[us]")
    qqq_ref["week_end"] = qqq_ref["week_end"].astype("datetime64[us]")
    merged = pd.merge_asof(
        merged.sort_values("week_end"),
        qqq_ref.sort_values("week_end"),
        on="week_end",
        direction="nearest",
        tolerance=pd.Timedelta(days=5),
    )

    # Write aligned dataset
    aligned_cols = ["week_end", "qqq_adj_close_weekly"] + REALIZED_METRICS
    for h in HORIZONS:
        for t in ["fwd_ret", "fwd_vol", "fwd_mdd"]:
            aligned_cols.append(f"{t}_{h}w")
    aligned_cols = [c for c in aligned_cols if c in merged.columns]
    merged[aligned_cols].to_csv(OUTPUT_DIR / "qqq_weekly_aligned.csv", index=False)
    print(f"[phase10] Written: qqq_weekly_aligned.csv ({len(merged)} rows)")

    # --- Part B: Regime separation ---
    print("[phase10] Running Part B: Regime Separation...")
    b_results = run_regime_separation(merged)

    # Write summary CSV
    summary_df = pd.DataFrame(b_results["summary_records"])
    summary_df.to_csv(OUTPUT_DIR / "regime_separation_summary.csv", index=False)

    # Write tests JSON
    with open(OUTPUT_DIR / "regime_separation_tests.json", "w") as f:
        json.dump(b_results["tests"], f, indent=2, default=_json_default)

    b_passed = b_results["passed"]
    print(f"[phase10] Part B passed: {b_passed}")

    # --- Part A: Signal predictive power ---
    print("[phase10] Running Part A: Signal Predictive Power...")
    high_conf = merged[merged["max_p_t"] >= HIGH_CONF_THRESHOLD].copy()
    a_df = run_signal_predictive(merged, high_conf)
    a_df.to_csv(OUTPUT_DIR / "signal_predictive_summary.csv", index=False)

    a_passed = check_signal_predictive_pass(a_df)
    print(f"[phase10] Part A passed: {a_passed}")

    # --- Acceptance document ---
    _write_acceptance(
        output_dir=OUTPUT_DIR,
        b_passed=b_passed,
        a_passed=a_passed,
        b_results=b_results,
        a_df=a_df,
        n_strict=len(merged),
        n_high_conf=len(high_conf),
        epoch_start=epoch_start,
        epoch_end=strict_last,
    )
    print("[phase10] Written: phase10_acceptance.md")
    print(f"[phase10] Phase 11 permitted: {b_passed and a_passed}")


def _load_qqq_daily(start: str, end: str, force_refresh: bool) -> pd.DataFrame:
    """Load QQQ daily adjusted close with local cache."""
    daily_cache = REPO_ROOT / "cache/phase10/qqq_daily_price.csv"

    if daily_cache.exists() and not force_refresh:
        df = pd.read_csv(daily_cache, parse_dates=["date"], index_col="date")
        return df

    import yfinance as yf

    fetch_start = (pd.Timestamp(start) - pd.DateOffset(weeks=12)).strftime("%Y-%m-%d")
    fetch_end = (pd.Timestamp(end) + pd.DateOffset(weeks=12)).strftime("%Y-%m-%d")

    raw = yf.download("QQQ", start=fetch_start, end=fetch_end, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("yfinance returned empty daily data for QQQ")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Close"]].rename(columns={"Close": "qqq_adj_close"})
    df.index.name = "date"
    daily_cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(daily_cache)
    return df


def _write_acceptance(
    output_dir: Path,
    b_passed: bool,
    a_passed: bool,
    b_results: dict,
    a_df: pd.DataFrame,
    n_strict: int,
    n_high_conf: int,
    epoch_start: str,
    epoch_end: str,
) -> None:
    phase11_permitted = b_passed and a_passed

    lines = [
        "# Phase 10 Acceptance",
        "",
        f"**Epoch**: `{epoch_start}` → `{epoch_end}`",
        f"**Strict rows**: {n_strict}  |  **High-confidence rows** (max_p_t ≥ 0.60): {n_high_conf}",
        "",
        "---",
        "",
        "## Part B — Regime Separation",
        "",
        f"**PASSED: {b_passed}**",
        "",
    ]

    for metric in REALIZED_METRICS:
        t = b_results["tests"].get(metric, {})
        kw_p = t.get("kw_pvalue", np.nan)
        n_sig_pairs = sum(p["significant"] for p in t.get("pairwise", []))
        lines.append(f"- `{metric}`: KW p={kw_p:.4f}, significant pairwise={n_sig_pairs}")

    lines += [
        "",
        "---",
        "",
        "## Part A — Signal Predictive Power",
        "",
        f"**PASSED: {a_passed}**",
        "",
        "Key results (full subsample, fwd_vol and fwd_mdd at h=4w and h=8w):",
        "",
        "| signal | horizon | target | spearman_rho | hac_pvalue | tercile_spread |",
        "|--------|---------|--------|-------------|------------|----------------|",
    ]

    for signal in ["rho_t", "h_t", "s_t"]:
        for h in [4, 8]:
            for tb in ["fwd_vol", "fwd_mdd"]:
                col = f"{tb}_{h}w"
                row = a_df[
                    (a_df["signal"] == signal)
                    & (a_df["horizon_weeks"] == h)
                    & (a_df["target"] == col)
                    & (a_df["subsample"] == "full")
                ]
                if row.empty:
                    continue
                r = row.iloc[0]
                sp = f"{r['spearman_rho']:.3f}" if not pd.isna(r["spearman_rho"]) else "nan"
                hp = f"{r['hac_pvalue']:.3f}" if not pd.isna(r["hac_pvalue"]) else "nan"
                ts = f"{r['tercile_spread']:.4f}" if not pd.isna(r["tercile_spread"]) else "nan"
                lines.append(f"| `{signal}` | {h}w | `{col}` | {sp} | {hp} | {ts} |")

    lines += [
        "",
        "---",
        "",
        "## Gate Decision",
        "",
        f"**Phase 11 permitted: {phase11_permitted}**",
        "",
    ]

    if not b_passed:
        lines.append(
            "> Part B failed: state labels do not correspond to distinct market environments. "
            "Do not proceed to strategy layer."
        )
    elif not a_passed:
        lines.append(
            "> Part B passed but Part A failed: signals carry insufficient predictive information. "
            "System is explanatory only. Do not proceed to investment layer."
        )
    else:
        lines.append(
            "> Both B and A passed. Regime separation confirmed. "
            "Signal predictive power confirmed. Phase 11 may proceed."
        )

    (output_dir / "phase10_acceptance.md").write_text("\n".join(lines) + "\n")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 10 analysis")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Re-download QQQ data ignoring cache")
    args = parser.parse_args()
    main(force_refresh=args.force_refresh)
