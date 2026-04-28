#!/usr/bin/env python
"""Run Phase 11 portfolio construction and P&L backtest."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qqq_cycle.backtest.engine import build_benchmark_nav, fetch_price_panel, run_backtest
from qqq_cycle.backtest.performance import PERFORMANCE_METRICS, compute_performance_metrics
from qqq_cycle.portfolio.construction import (
    BacktestConfig,
    build_weekly_weights,
    compute_s1_cluster_index,
)


DEFAULT_PIPELINE_CSV = REPO_ROOT / "outputs/pipeline/strict_real_pipeline_output.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/phase11"
DEFAULT_PRICE_CACHE = REPO_ROOT / "cache/phase11/price_panel_daily.csv"


def _load_strict_signals(pipeline_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(pipeline_csv, parse_dates=["week_end"])
    strict = df[(df["mode"] == "strict") & (df["strict_contracts_satisfied"] == True)].copy()  # noqa: E712
    if strict.empty:
        raise RuntimeError("no strict rows available for Phase 11")

    strict["rho_t"] = pd.to_numeric(strict["rho_t"], errors="coerce")
    strict["k_hat_t"] = pd.to_numeric(strict["k_hat_t"], errors="coerce")
    strict["drift_flag"] = strict["interpretability"].apply(_extract_drift_flag)
    strict = strict.sort_values("week_end", kind="mergesort").reset_index(drop=True)
    return strict[["week_end", "rho_t", "k_hat_t", "drift_flag", "interpretability", "I_t"]]


def _extract_drift_flag(raw: object) -> int:
    if not isinstance(raw, str) or not raw.strip():
        return 0
    parsed = json.loads(raw)
    return int(parsed.get("drift_flag", 0)) if isinstance(parsed, dict) else 0


def _write_manifest(output_dir: Path, signals: pd.DataFrame, s1_index: int) -> dict:
    drift_sum = int(signals["drift_flag"].sum())
    manifest = {
        "phase": "phase_11",
        "frozen_rules": {
            "assets": ["QQQ", "SHY"],
            "weight_rule": "omega_qqq = 1 - rho_t; omega_shy = rho_t",
            "rebalance_frequency": "weekly",
            "signal_observation_time": "friday_close",
            "execution_time": "next_trading_day_open",
            "transaction_cost_bps": 5.0,
            "transaction_cost_convention": "one_way_per_unit_turnover",
            "turnover_threshold": 0.05,
            "allow_leverage": False,
            "allow_short": False,
        },
        "circuit_breaker": {
            "trigger_condition": "k_hat_t == s1_cluster_index AND drift_flag == 1",
            "s1_cluster_index": int(s1_index),
            "s1_derivation_method": "median_H_I_partition_of_pipeline_output",
            "forced_weights": {"omega_qqq": 0.0, "omega_shy": 1.0},
            "release_condition": "2 consecutive weeks with k_hat_t != s1_cluster_index",
            "observed_drift_flag_sum": drift_sum,
        },
        "benchmarks": ["qqq_buyhold", "shy_buyhold", "static_6040"],
        "risk_free_rate": 0.0,
        "price_convention": "yfinance_auto_adjust_true_open_to_open",
        "epoch": {
            "start": signals["week_end"].min().strftime("%Y-%m-%d"),
            "end": signals["week_end"].max().strftime("%Y-%m-%d"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preregistration_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _benchmark_returns(price_panel: pd.DataFrame, omega_qqq: float, omega_shy: float) -> pd.Series:
    qqq = price_panel["exec_open_qqq"].shift(-1) / price_panel["exec_open_qqq"] - 1.0
    shy = price_panel["exec_open_shy"].shift(-1) / price_panel["exec_open_shy"] - 1.0
    returns = omega_qqq * qqq.iloc[:-1] + omega_shy * shy.iloc[:-1]
    returns.index = pd.to_datetime(price_panel["week_end"].iloc[:-1])
    return returns


def _write_performance_markdown(path: Path, summary: dict[str, dict[str, float]]) -> None:
    lines = ["# Phase 11 Performance Summary", ""]
    columns = ["strategy", "qqq_buyhold", "shy_buyhold", "static_6040"]
    lines.append("| metric | " + " | ".join(columns) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(columns)) + "|")
    for metric in PERFORMANCE_METRICS:
        values = []
        for col in columns:
            value = summary[col][metric]
            values.append("nan" if pd.isna(value) else f"{value:.10g}")
        lines.append(f"| {metric} | " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(
    *,
    pipeline_csv: str | Path = DEFAULT_PIPELINE_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    price_cache: str | Path = DEFAULT_PRICE_CACHE,
    force_refresh: bool = False,
) -> dict[str, Path]:
    """Run Phase 11 and write all preregistered artifacts.

    Input:
        pipeline_csv: Strict pipeline output CSV.
        output_dir: Artifact directory.
        price_cache: Cached QQQ/SHY adjusted-open panel.
        force_refresh: Re-download prices when true.

    Output:
        Mapping of artifact names to paths.

    Time/as-of semantics:
        Manifest is written immediately after loading strict signals and
        deriving S1, before price fetching, weight construction, or backtest
        return computation.
    """

    out = Path(output_dir)
    signals = _load_strict_signals(pipeline_csv)
    s1_index = compute_s1_cluster_index(signals)
    manifest = _write_manifest(out, signals, s1_index)

    config = BacktestConfig(
        transaction_cost_bps=manifest["frozen_rules"]["transaction_cost_bps"],
        turnover_threshold=manifest["frozen_rules"]["turnover_threshold"],
        circuit_breaker_s1_index=manifest["circuit_breaker"]["s1_cluster_index"],
        circuit_breaker_release_weeks=2,
    )

    price_panel = fetch_price_panel(
        manifest["epoch"]["start"],
        manifest["epoch"]["end"],
        price_cache,
        force_refresh=force_refresh,
    )

    weight_rows = build_weekly_weights(signals[["week_end", "rho_t", "k_hat_t", "drift_flag"]], config)
    weekly_weights = pd.DataFrame([asdict(row) for row in weight_rows])
    weekly_weights.to_csv(out / "weekly_weights.csv", index=False)

    backtest = run_backtest(weekly_weights, price_panel, config)
    backtest.to_csv(out / "backtest_returns.csv", index=False)

    qqq_nav = build_benchmark_nav(price_panel, 1.0, 0.0, config.transaction_cost_bps)
    shy_nav = build_benchmark_nav(price_panel, 0.0, 1.0, config.transaction_cost_bps)
    static_nav = build_benchmark_nav(price_panel, 0.60, 0.40, config.transaction_cost_bps)

    nav_comparison = pd.DataFrame(
        {
            "week_end": backtest["week_end"],
            "strategy": backtest["nav"],
            "qqq_buyhold": qqq_nav.to_numpy(dtype=float),
            "shy_buyhold": shy_nav.to_numpy(dtype=float),
            "static_6040": static_nav.to_numpy(dtype=float),
        }
    )
    nav_comparison.to_csv(out / "nav_comparison.csv", index=False)

    qqq_returns = _benchmark_returns(price_panel, 1.0, 0.0)
    shy_returns = _benchmark_returns(price_panel, 0.0, 1.0)
    static_returns = _benchmark_returns(price_panel, 0.60, 0.40)
    summary = {
        "strategy": compute_performance_metrics(
            backtest["net_portfolio_return"],
            rf=manifest["risk_free_rate"],
            turnover=backtest["turnover"],
            transaction_cost=backtest["transaction_cost"],
        ),
        "qqq_buyhold": compute_performance_metrics(qqq_returns, rf=manifest["risk_free_rate"]),
        "shy_buyhold": compute_performance_metrics(shy_returns, rf=manifest["risk_free_rate"]),
        "static_6040": compute_performance_metrics(static_returns, rf=manifest["risk_free_rate"]),
    }
    (out / "performance_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    _write_performance_markdown(out / "performance_summary.md", summary)

    artifacts = {
        "manifest": out / "preregistration_manifest.json",
        "weekly_weights": out / "weekly_weights.csv",
        "backtest_returns": out / "backtest_returns.csv",
        "nav_comparison": out / "nav_comparison.csv",
        "performance_summary_json": out / "performance_summary.json",
        "performance_summary_md": out / "performance_summary.md",
    }
    return artifacts


if __name__ == "__main__":
    written = main()
    for name, path in written.items():
        print(f"[phase11] {name}: {path}")
