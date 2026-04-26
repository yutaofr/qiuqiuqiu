"""Numerical-health summaries for replay diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _distribution(series: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {"count": 0, "p01": np.nan, "p05": np.nan, "p50": np.nan, "p95": np.nan, "p99": np.nan}
    qs = np.quantile(values, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        "count": int(len(values)),
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p50": float(qs[2]),
        "p95": float(qs[3]),
        "p99": float(qs[4]),
    }


def summarize_numerical_health(replay: pd.DataFrame) -> dict[str, Any]:
    """Return machine-readable numerical health distribution summary."""

    rows = len(replay)
    distributions = {
        metric: _distribution(replay[metric])
        for metric in [
            "maha",
            "huber_weight",
            "condition_number_raw",
            "condition_number_reg",
        ]
    }
    warmup = int((~replay["is_warm"].astype(bool)).sum())
    warm = int(replay["is_warm"].astype(bool).sum())
    return {
        "counts": {
            "rows": int(rows),
            "drift_flag_count": int(pd.to_numeric(replay["drift_flag"], errors="coerce").fillna(0).sum()),
        },
        "coverage": {
            "warmup_rows": warmup,
            "warm_rows": warm,
        },
        "distributions": distributions,
        "frequencies": {
            "eigval_2_was_floored_frequency": float(replay["eigval_2_was_floored"].fillna(False).astype(bool).mean()),
            "state_health_degradation_frequency": float((~replay["state_ok"].fillna(False).astype(bool)).mean()),
            "huber_weight_lt_1_frequency": float((pd.to_numeric(replay["huber_weight"], errors="coerce") < 1.0).mean()),
        },
    }


def build_tail_diagnostics(replay: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return required tail diagnostic extracts from a replay table."""

    frame = replay.copy()
    cond = pd.to_numeric(frame["condition_number_reg"], errors="coerce")
    huber = pd.to_numeric(frame["huber_weight"], errors="coerce")
    warm = frame["is_warm"].astype(bool)
    warm_indices = list(frame.index[warm])
    if warm_indices:
        boundary = int(warm_indices[0])
        start = max(0, boundary - 10)
        end = min(len(frame), boundary + 11)
        warmup_boundary = frame.iloc[start:end].copy()
    else:
        warmup_boundary = frame.iloc[0:0].copy()
    return {
        "top_20_condition_number_reg": frame.loc[cond.sort_values(ascending=False).head(20).index].copy(),
        "bottom_20_huber_weight": frame.loc[huber.sort_values(ascending=True).head(20).index].copy(),
        "drift_flags": frame[pd.to_numeric(frame["drift_flag"], errors="coerce").fillna(0).astype(int) == 1].copy(),
        "warmup_boundary_pm10": warmup_boundary,
    }


def write_tail_diagnostics(replay: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Write required tail diagnostic CSV files."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in build_tail_diagnostics(replay).items():
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


def write_health_summary(summary: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Write JSON and Markdown health summaries."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "numerical_health_summary.json"
    md_path = out / "numerical_health_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Numerical Health Summary",
        "",
        f"- Rows: {summary['counts']['rows']}",
        f"- Warmup rows: {summary['coverage']['warmup_rows']}",
        f"- Warm rows: {summary['coverage']['warm_rows']}",
        f"- Drift flag count: {summary['counts']['drift_flag_count']}",
        f"- Eigval floor frequency: {summary['frequencies']['eigval_2_was_floored_frequency']:.6f}",
        f"- State health degradation frequency: {summary['frequencies']['state_health_degradation_frequency']:.6f}",
        f"- Huber weight < 1 frequency: {summary['frequencies']['huber_weight_lt_1_frequency']:.6f}",
        "",
        "## Distributions",
    ]
    for metric, dist in summary["distributions"].items():
        lines.append(
            f"- {metric}: count={dist['count']}, p01={dist['p01']:.6g}, "
            f"p05={dist['p05']:.6g}, p50={dist['p50']:.6g}, "
            f"p95={dist['p95']:.6g}, p99={dist['p99']:.6g}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
