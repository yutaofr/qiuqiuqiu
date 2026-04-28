"""Tests for Phase 11 performance metrics and artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qqq_cycle.backtest.performance import (
    PERFORMANCE_METRICS,
    build_comparison_table,
    compute_performance_metrics,
)
from scripts.run_phase11 import main as run_phase11_main


def test_performance_summary_contains_all_metrics() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01], name="net_portfolio_return")

    metrics = compute_performance_metrics(returns)

    assert set(PERFORMANCE_METRICS).issubset(metrics.keys())


def test_benchmarks_present() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, 0.0])

    table = build_comparison_table(returns, returns, returns, returns)

    assert list(table.columns) == ["strategy", "qqq_buyhold", "shy_buyhold", "static_6040"]


def test_output_artifacts_exist(tmp_path: Path) -> None:
    pipeline_csv = tmp_path / "pipeline.csv"
    cache_path = tmp_path / "price_panel_daily.csv"
    output_dir = tmp_path / "phase11"

    _write_pipeline_fixture(pipeline_csv)
    _write_price_cache(cache_path)

    run_phase11_main(
        pipeline_csv=pipeline_csv,
        output_dir=output_dir,
        price_cache=cache_path,
        force_refresh=False,
    )

    expected = {
        "preregistration_manifest.json",
        "weekly_weights.csv",
        "backtest_returns.csv",
        "nav_comparison.csv",
        "performance_summary.json",
        "performance_summary.md",
    }
    assert expected == {p.name for p in output_dir.iterdir()}

    summary = json.loads((output_dir / "performance_summary.json").read_text())
    assert set(PERFORMANCE_METRICS).issubset(summary["strategy"].keys())


def test_strategy_and_benchmark_use_same_calendar(tmp_path: Path) -> None:
    pipeline_csv = tmp_path / "pipeline.csv"
    cache_path = tmp_path / "price_panel_daily.csv"
    output_dir = tmp_path / "phase11"

    _write_pipeline_fixture(pipeline_csv)
    _write_price_cache(cache_path)

    run_phase11_main(
        pipeline_csv=pipeline_csv,
        output_dir=output_dir,
        price_cache=cache_path,
        force_refresh=False,
    )

    nav = pd.read_csv(output_dir / "nav_comparison.csv")
    calendars = [
        nav.loc[nav[col].notna(), "week_end"].tolist()
        for col in ["strategy", "qqq_buyhold", "shy_buyhold", "static_6040"]
    ]
    assert calendars[1:] == [calendars[0], calendars[0], calendars[0]]


def _write_pipeline_fixture(path: Path) -> None:
    week_ends = pd.date_range("2024-01-05", periods=10, freq="W-FRI")
    h_by_cluster = {0: -2.0, 1: -1.4, 2: 0.1, 3: 0.4, 4: 0.8}
    i_by_cluster = {0: -0.8, 1: 0.4, 2: 0.0, 3: -0.2, 4: 0.3}
    rows = []
    for i, week_end in enumerate(week_ends):
        cluster = i % 5
        interp = {
            "H": h_by_cluster[cluster] + 0.01 * i,
            "I": i_by_cluster[cluster],
            "drift_flag": 0,
        }
        rows.append(
            {
                "week_end": week_end.strftime("%Y-%m-%d"),
                "k_hat_t": cluster,
                "p_t": json.dumps([0.2] * 5),
                "s_t": 0.1,
                "h_t": 0.2,
                "rho_t": np.clip(0.2 + 0.05 * i, 0.0, 1.0),
                "I_t": i_by_cluster[cluster],
                "interpretability": json.dumps(interp),
                "mode": "strict",
                "degraded_reason": "",
                "strict_contracts_satisfied": True,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_price_cache(path: Path) -> None:
    week_ends = pd.date_range("2024-01-05", periods=10, freq="W-FRI")
    exec_dates = week_ends + pd.offsets.BDay(1)
    panel = pd.DataFrame(
        {
            "week_end": week_ends,
            "exec_date": exec_dates,
            "exec_open_qqq": 100.0 * (1.01 ** np.arange(len(week_ends))),
            "exec_open_shy": 100.0 * (1.001 ** np.arange(len(week_ends))),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(path, index=False)
