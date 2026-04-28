"""Phase 14 latest-view regime monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from qqq_cycle.ops.revision_audit import RevisionAuditInputError, load_snapshot_history


@dataclass(frozen=True)
class RegimeMonitorArtifacts:
    """Paths written by one regime-monitor run."""

    transition_matrix_path: Path
    duration_summary_path: Path
    event_response_summary_path: Path


def load_latest_snapshot_per_week(history_dir: str | Path) -> pd.DataFrame:
    """Load the latest immutable snapshot for each week_end.

    Input: Phase 14 immutable history directory. Output: one row per week_end,
    chosen by the latest published_at timestamp within that week, sorted by
    decision week_end. This is the canonical latest-view loader reused by 14C/14D.
    """

    history = load_snapshot_history(history_dir)
    latest = (
        history.sort_values(["week_end", "published_at", "snapshot_path"], kind="mergesort")
        .groupby("week_end", sort=True, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    latest["execution_state"] = latest["payload"].apply(lambda payload: payload.get("execution_state"))
    latest["execution_permitted"] = latest["payload"].apply(lambda payload: payload.get("execution_permitted"))
    latest["signal_valid_but_not_executable"] = latest["payload"].apply(
        lambda payload: payload.get("signal_valid_but_not_executable")
    )
    latest["strict_contracts_satisfied"] = latest["payload"].apply(
        lambda payload: payload.get("strict_contracts_satisfied")
    )
    latest["degraded_reason"] = latest["payload"].apply(lambda payload: payload.get("degraded_reason"))
    latest["execution_block_reason"] = latest["payload"].apply(
        lambda payload: payload.get("execution_block_reason")
    )
    latest["week_end_ts"] = pd.to_datetime(latest["week_end"])
    latest["regime_label"] = latest.apply(_regime_label, axis=1)
    latest["published_at_text"] = latest["published_at"].apply(_ts_text)
    return latest.sort_values("week_end_ts", kind="mergesort").reset_index(drop=True)


def build_state_transition_matrix(latest_view: pd.DataFrame) -> pd.DataFrame:
    """Build a wide transition-count matrix from consecutive latest-view weeks."""

    labels = latest_view["regime_label"].dropna().astype(str).tolist()
    unique_labels = list(dict.fromkeys(labels))
    matrix = pd.DataFrame(0, index=unique_labels, columns=unique_labels, dtype=int)
    matrix.index.name = "from_regime"
    matrix.columns.name = "to_regime"

    if len(latest_view) < 2:
        return matrix

    previous = latest_view["regime_label"].iloc[:-1].astype(str).tolist()
    current = latest_view["regime_label"].iloc[1:].astype(str).tolist()
    for from_label, to_label in zip(previous, current, strict=True):
        matrix.loc[from_label, to_label] += 1
    return matrix


def build_state_duration_summary(latest_view: pd.DataFrame) -> pd.DataFrame:
    """Summarize contiguous latest-view regime durations."""

    runs = _build_regime_runs(latest_view)
    if not runs:
        return pd.DataFrame(
            columns=[
                "regime_label",
                "mode",
                "k_hat_t",
                "run_count",
                "min_duration_weeks",
                "median_duration_weeks",
                "mean_duration_weeks",
                "max_duration_weeks",
                "latest_run_start",
                "latest_run_end",
                "latest_run_duration_weeks",
            ]
        )

    runs_frame = pd.DataFrame(runs)
    summary = (
        runs_frame.groupby(["regime_label", "mode", "k_hat_t"], dropna=False, sort=True)
        .agg(
            run_count=("duration_weeks", "size"),
            min_duration_weeks=("duration_weeks", "min"),
            median_duration_weeks=("duration_weeks", "median"),
            mean_duration_weeks=("duration_weeks", "mean"),
            max_duration_weeks=("duration_weeks", "max"),
        )
        .reset_index()
    )

    latest_runs = (
        runs_frame.sort_values("end_week_end", kind="mergesort")
        .groupby("regime_label", sort=False, as_index=False)
        .tail(1)[["regime_label", "start_week_end", "end_week_end", "duration_weeks"]]
        .rename(
            columns={
                "start_week_end": "latest_run_start",
                "end_week_end": "latest_run_end",
                "duration_weeks": "latest_run_duration_weeks",
            }
        )
    )
    merged = summary.merge(latest_runs, on="regime_label", how="left")
    merged["mean_duration_weeks"] = merged["mean_duration_weeks"].astype(float)
    merged["median_duration_weeks"] = merged["median_duration_weeks"].astype(float)
    return merged.sort_values(["regime_label"], kind="mergesort").reset_index(drop=True)


def build_event_response_summary(latest_view: pd.DataFrame) -> pd.DataFrame:
    """Summarize changes in s_t, h_t, and rho_t around regime transitions."""

    events: list[dict[str, Any]] = []
    if len(latest_view) < 2:
        return pd.DataFrame(
            columns=[
                "from_regime",
                "to_regime",
                "event_count",
                "mean_pre_s_t",
                "mean_post_s_t",
                "mean_delta_s_t",
                "mean_pre_h_t",
                "mean_post_h_t",
                "mean_delta_h_t",
                "mean_pre_rho_t",
                "mean_post_rho_t",
                "mean_delta_rho_t",
            ]
        )

    for previous, current in zip(latest_view.iloc[:-1].to_dict("records"), latest_view.iloc[1:].to_dict("records"), strict=True):
        if previous["regime_label"] == current["regime_label"]:
            continue
        events.append(
            {
                "transition_week_end": current["week_end"],
                "from_regime": previous["regime_label"],
                "to_regime": current["regime_label"],
                "pre_s_t": previous.get("s_t"),
                "post_s_t": current.get("s_t"),
                "delta_s_t": _delta(previous.get("s_t"), current.get("s_t")),
                "pre_h_t": previous.get("h_t"),
                "post_h_t": current.get("h_t"),
                "delta_h_t": _delta(previous.get("h_t"), current.get("h_t")),
                "pre_rho_t": previous.get("rho_t"),
                "post_rho_t": current.get("rho_t"),
                "delta_rho_t": _delta(previous.get("rho_t"), current.get("rho_t")),
            }
        )

    if not events:
        return pd.DataFrame(
            columns=[
                "from_regime",
                "to_regime",
                "event_count",
                "mean_pre_s_t",
                "mean_post_s_t",
                "mean_delta_s_t",
                "mean_pre_h_t",
                "mean_post_h_t",
                "mean_delta_h_t",
                "mean_pre_rho_t",
                "mean_post_rho_t",
                "mean_delta_rho_t",
            ]
        )

    event_frame = pd.DataFrame(events)
    summary = (
        event_frame.groupby(["from_regime", "to_regime"], sort=True)
        .agg(
            event_count=("transition_week_end", "size"),
            mean_pre_s_t=("pre_s_t", "mean"),
            mean_post_s_t=("post_s_t", "mean"),
            mean_delta_s_t=("delta_s_t", "mean"),
            mean_pre_h_t=("pre_h_t", "mean"),
            mean_post_h_t=("post_h_t", "mean"),
            mean_delta_h_t=("delta_h_t", "mean"),
            mean_pre_rho_t=("pre_rho_t", "mean"),
            mean_post_rho_t=("post_rho_t", "mean"),
            mean_delta_rho_t=("delta_rho_t", "mean"),
        )
        .reset_index()
    )
    return summary


def write_regime_monitor_outputs(
    *,
    history_dir: str | Path,
    output_dir: str | Path,
) -> RegimeMonitorArtifacts:
    """Run latest-view regime monitoring and write CSV artifacts."""

    latest_view = load_latest_snapshot_per_week(history_dir)
    transition_matrix = build_state_transition_matrix(latest_view)
    duration_summary = build_state_duration_summary(latest_view)
    event_response_summary = build_event_response_summary(latest_view)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    transition_matrix_path = out / "state_transition_matrix.csv"
    duration_summary_path = out / "state_duration_summary.csv"
    event_response_summary_path = out / "event_response_summary.csv"

    transition_matrix.to_csv(transition_matrix_path)
    duration_summary.to_csv(duration_summary_path, index=False)
    event_response_summary.to_csv(event_response_summary_path, index=False)

    return RegimeMonitorArtifacts(
        transition_matrix_path=transition_matrix_path,
        duration_summary_path=duration_summary_path,
        event_response_summary_path=event_response_summary_path,
    )


def _build_regime_runs(latest_view: pd.DataFrame) -> list[dict[str, Any]]:
    if latest_view.empty:
        return []

    rows = latest_view.sort_values("week_end_ts", kind="mergesort").reset_index(drop=True)
    runs: list[dict[str, Any]] = []
    current_label = None
    current_start = None
    current_mode = None
    current_k_hat = None
    duration = 0

    for row in rows.to_dict("records"):
        label = row["regime_label"]
        if current_label is None:
            current_label = label
            current_start = row["week_end"]
            current_mode = row["mode"]
            current_k_hat = row["k_hat_t"]
            duration = 1
            last_week_end = row["week_end"]
            continue

        if label == current_label:
            duration += 1
            last_week_end = row["week_end"]
            continue

        runs.append(
            {
                "regime_label": current_label,
                "mode": current_mode,
                "k_hat_t": current_k_hat,
                "start_week_end": current_start,
                "end_week_end": last_week_end,
                "duration_weeks": duration,
            }
        )
        current_label = label
        current_start = row["week_end"]
        current_mode = row["mode"]
        current_k_hat = row["k_hat_t"]
        duration = 1
        last_week_end = row["week_end"]

    runs.append(
        {
            "regime_label": current_label,
            "mode": current_mode,
            "k_hat_t": current_k_hat,
            "start_week_end": current_start,
            "end_week_end": last_week_end,
            "duration_weeks": duration,
        }
    )
    return runs


def _regime_label(row: pd.Series) -> str:
    mode = str(row.get("mode"))
    k_hat_t = row.get("k_hat_t")
    if k_hat_t is None or pd.isna(k_hat_t):
        return mode
    return f"{mode}:k{int(k_hat_t)}"


def _delta(initial: Any, latest: Any) -> float | None:
    if initial is None or latest is None or pd.isna(initial) or pd.isna(latest):
        return None
    return float(latest) - float(initial)


def _ts_text(value: pd.Timestamp) -> str:
    if value.tz is None:
        raise RevisionAuditInputError("published_at must be timezone-aware")
    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")
