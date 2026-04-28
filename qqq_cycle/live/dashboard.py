"""Dashboard-ready flat CSV exporters for the live interpretability layer.

Each writer appends one row per weekly run to its respective CSV file.
Files are created on first write. Column headers are written only when
the file does not yet exist.

Output files (all in output_dir/):
    dashboard_state_plane.csv     — H/I position + cluster centroids
    dashboard_drift_monitor.csv   — drift probe value + flag + thresholds
    dashboard_pollution_flags.csv — per-source freshness + execution tier
"""

from __future__ import annotations

import csv
from pathlib import Path

from qqq_cycle.live.interpretability import InterpretabilitySnapshot

_N_CLUSTERS = 5   # number of prototype clusters


# ---------------------------------------------------------------------------
# State plane
# ---------------------------------------------------------------------------

def _state_plane_fieldnames() -> list[str]:
    centroid_cols = [
        f"centroid_{k}_{dim}"
        for k in range(_N_CLUSTERS)
        for dim in ("H", "I")
    ]
    return [
        "week_end", "H_t", "I_t", "k_hat_t", "state_label",
        *[f"p_{k}" for k in range(_N_CLUSTERS)],
        *centroid_cols,
    ]


def append_state_plane(snap: InterpretabilitySnapshot, output_dir: Path) -> None:
    """Append one row to dashboard_state_plane.csv."""
    path = output_dir / "dashboard_state_plane.csv"
    fieldnames = _state_plane_fieldnames()
    row: dict = {
        "week_end": snap.week_end,
        "H_t": snap.H_t,
        "I_t": snap.I_t,
        "k_hat_t": snap.k_hat_t,
        "state_label": snap.state_label,
    }
    probs = snap.state_probabilities or {}
    for k in range(_N_CLUSTERS):
        row[f"p_{k}"] = probs.get(str(k))

    centroids = snap.centroids or []
    for k in range(_N_CLUSTERS):
        if k < len(centroids) and len(centroids[k]) >= 2:
            row[f"centroid_{k}_H"] = centroids[k][0]
            row[f"centroid_{k}_I"] = centroids[k][1]
        else:
            row[f"centroid_{k}_H"] = None
            row[f"centroid_{k}_I"] = None

    _append_row(path, fieldnames, row)


# ---------------------------------------------------------------------------
# Drift monitor
# ---------------------------------------------------------------------------

_DRIFT_FIELDNAMES = [
    "week_end",
    "drift_probe_raw",
    "drift_flag",
    "threshold_lo",
    "threshold_hi",
]


def append_drift_monitor(snap: InterpretabilitySnapshot, output_dir: Path) -> None:
    """Append one row to dashboard_drift_monitor.csv."""
    path = output_dir / "dashboard_drift_monitor.csv"
    dm = snap.drift_metrics
    row = {
        "week_end": snap.week_end,
        "drift_probe_raw": dm.get("drift_probe_raw"),
        "drift_flag": dm.get("drift_flag"),
        "threshold_lo": dm.get("threshold_lo"),
        "threshold_hi": dm.get("threshold_hi"),
    }
    _append_row(path, _DRIFT_FIELDNAMES, row)


# ---------------------------------------------------------------------------
# Pollution flags
# ---------------------------------------------------------------------------

_POLLUTION_FIXED_COLS = ["week_end", "execution_tier", "stale_sources"]
_KNOWN_SOURCES = [
    "fred_macro", "ai_gpr", "qqq_prices", "constituents",
    "weights", "pit_engine",
]


def _pollution_fieldnames() -> list[str]:
    return [
        *_POLLUTION_FIXED_COLS,
        *[f"{s}_fresh" for s in _KNOWN_SOURCES],
    ]


def append_pollution_flags(snap: InterpretabilitySnapshot, output_dir: Path) -> None:
    """Append one row to dashboard_pollution_flags.csv."""
    path = output_dir / "dashboard_pollution_flags.csv"
    fieldnames = _pollution_fieldnames()
    pf = snap.pollution_flags
    stale = pf.get("stale_sources", [])
    row: dict = {
        "week_end": snap.week_end,
        "execution_tier": snap.execution_tier,
        "stale_sources": "|".join(stale) if stale else "",
    }
    for s in _KNOWN_SOURCES:
        row[f"{s}_fresh"] = pf.get(f"{s}_fresh")

    _append_row(path, fieldnames, row)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _append_row(path: Path, fieldnames: list[str], row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
