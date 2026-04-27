"""State/stress replay audit helpers.

This module audits existing diagnostic replay artifacts. It does not compute
microstructure fragility, production risk, returns, corporate actions, h_t, or
rho_t. Inputs are treated as already generated point-in-time state/stress replay
tables, and all comparisons are window-local diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from qqq_cycle.backtest.diagnostics import EVENT_WINDOWS, build_replay_bundle

LOW_HEAT_STATES = frozenset({"S1", "S2"})
EVENT_AUDIT_COLUMNS = [
    "window",
    "start",
    "end",
    "rows_total",
    "rows_finite_H_t",
    "rows_finite_s_t",
    "dominant_state_sequence",
    "first_week_low_heat_states_become_material",
    "first_week_s_t_breaks_into_upper_tail_regime",
    "stress_upper_tail_threshold",
    "drift_flag_rows",
    "lag_weeks_between_state_migration_and_stress_breakout",
    "state_label_status",
    "state_label_blocking_stage",
    "state_label_earliest_valid_date",
    "state_label_blocking_reason",
]
RAW_INPUT_COLUMNS = [
    "DFII10",
    "DGS2",
    "BAMLH0A0HYM2",
    "NFCI",
    "VIXCLS",
    "USEPUINDXD",
    "AI_GPR",
    "QQQ",
]


def sha256_file(path: str | Path) -> str:
    """Return SHA-256 for a local file."""

    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _compressed_sequence(labels: pd.Series) -> str:
    clean = labels.dropna().astype(str)
    clean = clean[clean != ""]
    if clean.empty:
        return ""
    out: list[str] = []
    previous: str | None = None
    for label in clean:
        if label != previous:
            out.append(label)
            previous = label
    return " -> ".join(out)


def _first_date(frame: pd.DataFrame, mask: pd.Series) -> str | None:
    if not bool(mask.any()):
        return None
    return str(frame.loc[mask, "week_end"].iloc[0])


def _week_lag(first: str | None, second: str | None) -> int | None:
    if first is None or second is None:
        return None
    return int(round((pd.Timestamp(second) - pd.Timestamp(first)).days / 7))


def _date_or_none(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _first_finite(series: pd.Series) -> str | None:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[values.notna()]
    if finite.empty:
        return None
    return _date_or_none(finite.index[0])


def _first_matching_date(frame: pd.DataFrame, mask: pd.Series) -> str | None:
    if frame.empty or not bool(mask.any()):
        return None
    return _date_or_none(pd.to_datetime(frame.loc[mask, "week_end"]).iloc[0])


def _nth_finite_date(theta: pd.DataFrame, n: int) -> str | None:
    finite = theta.dropna()
    if len(finite) < n:
        return None
    return _date_or_none(finite.index[n - 1])


def build_warmup_dependency_map(
    weekly_inputs: pd.DataFrame,
    replay: pd.DataFrame,
    *,
    raw_first_dates: Mapping[str, str | None] | None = None,
) -> pd.DataFrame:
    """Build a stage-by-stage state/stress warmup dependency map.

    Inputs:
        weekly_inputs: Weekly replay inputs indexed by week end, or containing a
            `week_end` column.
        replay: Generated weekly replay table.
        raw_first_dates: Optional raw-source first dates from cache files.

    Output:
        A table with first raw, transformed, weekly, state-probability, and
        stress usability dates for each required stage.

    Time semantics:
        This function diagnoses already generated weekly artifacts. It does not
        impute values or alter state/stress outputs.
    """

    from qqq_cycle.core.drift_probe import DriftProbe
    from qqq_cycle.core.state_layer import compute_state_layer
    from qqq_cycle.core.stress_layer import compute_stress_layer

    inputs = weekly_inputs.copy()
    if "week_end" in inputs.columns:
        inputs = inputs.set_index(pd.to_datetime(inputs["week_end"])).drop(columns=["week_end"])
    else:
        inputs.index = pd.to_datetime(inputs.index)
    state = compute_state_layer(inputs)
    theta = state[["H", "I"]]
    stress = compute_stress_layer(theta, state["E"]).frame
    drift = DriftProbe().compute(inputs)
    replay_frame = replay.copy()
    replay_frame["week_end"] = pd.to_datetime(replay_frame["week_end"])

    raw_first_dates = raw_first_dates or {}
    first_state_label = _first_matching_date(
        replay_frame, replay_frame["state_label"].astype(str) != "WARMUP"
    )
    first_cov_warm = _first_matching_date(
        replay_frame, replay_frame["is_warm"].fillna(False).astype(bool)
    )
    first_theta = _first_finite(theta["H"].where(theta["I"].notna()))
    theta_260 = _nth_finite_date(theta, 260)
    first_stress = _first_finite(stress["s"])

    rows: list[dict[str, object]] = []

    def add(
        stage: str,
        *,
        first_available_raw_date: str | None = None,
        first_finite_transformed_date: str | None = None,
        first_finite_weekly_date: str | None = None,
        first_date_usable_for_state_probability: str | None = None,
        first_date_usable_for_stress: str | None = None,
        blocking_reason: str | None = None,
    ) -> None:
        rows.append(
            {
                "stage": stage,
                "first_available_raw_date": first_available_raw_date,
                "first_finite_transformed_date": first_finite_transformed_date,
                "first_finite_weekly_date": first_finite_weekly_date,
                "first_date_usable_for_state_probability": first_date_usable_for_state_probability,
                "first_date_usable_for_stress": first_date_usable_for_stress,
                "blocking_reason": blocking_reason,
            }
        )

    for column in RAW_INPUT_COLUMNS:
        add(
            column,
            first_available_raw_date=raw_first_dates.get(column) or _first_finite(inputs[column]),
            first_finite_weekly_date=_first_finite(inputs[column]),
        )

    transformed = {
        "L_t": state["L"],
        "T_t": state["T"],
        "P_t": state["P"],
        "E_t": state["E"],
        "H_t": state["H"],
        "I_t": state["I"],
        "d_t": stress["d"],
        "a_t": stress["a"],
        "s_t": stress["s"],
        "drift_probe_raw": drift["drift_probe_raw"],
    }
    for stage, series in transformed.items():
        first = _first_finite(series)
        add(
            stage,
            first_finite_transformed_date=first,
            first_finite_weekly_date=first,
            first_date_usable_for_state_probability=first_state_label
            if stage in {"H_t", "I_t"}
            else None,
            first_date_usable_for_stress=first_stress if stage in {"H_t", "I_t", "E_t"} else first
            if stage in {"d_t", "a_t", "s_t"}
            else None,
        )

    add(
        "covariance_warmup",
        first_finite_transformed_date=first_theta,
        first_date_usable_for_state_probability=first_cov_warm,
        blocking_reason="requires 260 finite Theta(H,I) updates; NaN Theta rows do not advance warmup_count",
    )
    add(
        "prototype_state_assignment_warmup",
        first_finite_transformed_date=theta_260,
        first_date_usable_for_state_probability=first_state_label,
        blocking_reason=(
            "requires initialized prototypes after the 260 finite-Theta covariance "
            "warmup; current replay emits the first non-WARMUP label on the next finite week"
        ),
    )
    add(
        "state_label",
        first_date_usable_for_state_probability=first_state_label,
        blocking_reason="locked until covariance warmup and prototype initialization complete",
    )
    return pd.DataFrame(rows)


def explain_warmup_boundary(
    warmup_map: pd.DataFrame,
    *,
    window_name: str,
    start: str,
    end: str,
) -> dict[str, object]:
    """Explain whether an event window is blocked by state-label warmup."""

    lookup = warmup_map.set_index("stage")
    first_theta = lookup.at["covariance_warmup", "first_finite_transformed_date"]
    covariance_usable = lookup.at[
        "covariance_warmup", "first_date_usable_for_state_probability"
    ]
    state_label_usable = lookup.at["state_label", "first_date_usable_for_state_probability"]
    window_end = pd.Timestamp(end)
    blocked = state_label_usable is not None and window_end < pd.Timestamp(state_label_usable)
    binding_stage = (
        "covariance_warmup" if blocked and window_end < pd.Timestamp(covariance_usable) else None
    )
    if blocked and binding_stage is None:
        binding_stage = "prototype_state_assignment_warmup"
    reason = None
    if blocked:
        reason = (
            "The event window ends before state probabilities and semantic labels "
            "are unlocked. The binding chain is raw weekly inputs -> dual-memory "
            "state factors -> finite Theta(H,I) -> 260 finite-Theta covariance "
            "warmup -> prototype initialization/state assignment."
        )
    return {
        "window": window_name,
        "start": start,
        "end": end,
        "is_blocked": bool(blocked),
        "binding_stage": binding_stage,
        "first_finite_theta_date": first_theta,
        "covariance_earliest_usable_date": covariance_usable,
        "state_label_earliest_valid_date": state_label_usable,
        "dependency_chain": [
            "raw weekly inputs",
            "dual-memory factors L_t/T_t/P_t",
            "Theta(H_t,I_t)",
            "260 finite-Theta covariance warmup",
            "prototype initialization/state assignment",
            "state_label",
        ],
        "blocking_reason": reason,
    }


def write_warmup_explanation(
    warmup_map: pd.DataFrame,
    explanation: Mapping[str, object],
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Write machine- and human-readable warmup boundary artifacts."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    map_path = out / "warmup_dependency_map.csv"
    json_path = out / "warmup_explanation.json"
    md_path = out / "warmup_explanation.md"
    warmup_map.to_csv(map_path, index=False)
    json_path.write_text(json.dumps(explanation, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Warmup Explanation",
        "",
        f"- Window: {explanation['window']} ({explanation['start']} to {explanation['end']})",
        f"- Blocked: {explanation['is_blocked']}",
        f"- Binding stage: {explanation['binding_stage']}",
        f"- First finite Theta date: {explanation['first_finite_theta_date']}",
        f"- Covariance earliest usable date: {explanation['covariance_earliest_usable_date']}",
        f"- State-label earliest valid date: {explanation['state_label_earliest_valid_date']}",
        "",
        "## Dependency Chain",
    ]
    lines.extend(f"- {item}" for item in explanation["dependency_chain"])
    lines.extend(["", "## Blocking Reason", "", str(explanation["blocking_reason"])])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return map_path, json_path, md_path


def summarize_behavior_window(
    replay: pd.DataFrame,
    *,
    window_name: str,
    start: str,
    end: str,
    stress_upper_tail_threshold: float,
    warmup_explanation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Summarize state/stress behavior in one event window.

    Inputs:
        replay: Weekly diagnostic replay table with `week_end`, `H_t`, `s_t`,
            `state_label`, and `drift_flag`.
        window_name/start/end: Closed event-window bounds.
        stress_upper_tail_threshold: Full-sample finite `s_t` threshold used to
            identify the first upper-tail stress breakout.

    Output:
        A single audit row. Low-heat state migration is based only on semantic
        labels S1/S2; WARMUP rows are not treated as material state migration.

    Time semantics:
        The function reads only replay rows already inside or before the window
        table and never changes model outputs.
    """

    frame = replay.copy()
    frame["week_end"] = pd.to_datetime(frame["week_end"])
    window = frame[
        (frame["week_end"] >= pd.Timestamp(start))
        & (frame["week_end"] <= pd.Timestamp(end))
    ].copy()
    window["week_end"] = window["week_end"].dt.strftime("%Y-%m-%d")
    h = pd.to_numeric(window.get("H_t"), errors="coerce")
    s = pd.to_numeric(window.get("s_t"), errors="coerce")
    labels = window.get("state_label", pd.Series(dtype=object)).astype(str)
    low_heat_mask = labels.isin(LOW_HEAT_STATES)
    stress_mask = s >= stress_upper_tail_threshold
    drift = pd.to_numeric(window.get("drift_flag"), errors="coerce").fillna(0).astype(int)

    first_low = _first_date(window, low_heat_mask)
    first_stress = _first_date(window, stress_mask)
    state_label_status = "valid"
    blocking_stage = None
    earliest_valid = None
    blocking_reason = None
    if warmup_explanation is not None and bool(warmup_explanation.get("is_blocked")):
        state_label_status = "mathematically_blocked"
        blocking_stage = warmup_explanation.get("binding_stage")
        earliest_valid = warmup_explanation.get("state_label_earliest_valid_date")
        blocking_reason = warmup_explanation.get("blocking_reason")
    return {
        "window": window_name,
        "start": start,
        "end": end,
        "rows_total": int(len(window)),
        "rows_finite_H_t": int(h.notna().sum()),
        "rows_finite_s_t": int(s.notna().sum()),
        "dominant_state_sequence": _compressed_sequence(labels),
        "first_week_low_heat_states_become_material": first_low,
        "first_week_s_t_breaks_into_upper_tail_regime": first_stress,
        "stress_upper_tail_threshold": float(stress_upper_tail_threshold),
        "drift_flag_rows": int((drift == 1).sum()),
        "lag_weeks_between_state_migration_and_stress_breakout": _week_lag(
            first_low, first_stress
        ),
        "state_label_status": state_label_status,
        "state_label_blocking_stage": blocking_stage,
        "state_label_earliest_valid_date": earliest_valid,
        "state_label_blocking_reason": blocking_reason,
    }


def write_behavior_audits(
    replay: pd.DataFrame,
    output_dir: str | Path,
    windows: Mapping[str, tuple[str, str]] = EVENT_WINDOWS,
    warmup_explanations: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Path]:
    """Write one behavior audit CSV per event window plus a combined table."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    threshold = float(pd.to_numeric(replay["s_t"], errors="coerce").quantile(0.80))
    paths: dict[str, Path] = {}
    rows: list[dict[str, object]] = []
    for name, (start, end) in windows.items():
        row = summarize_behavior_window(
            replay,
            window_name=name,
            start=start,
            end=end,
            stress_upper_tail_threshold=threshold,
            warmup_explanation=(warmup_explanations or {}).get(name),
        )
        rows.append(row)
        frame = pd.DataFrame([row], columns=EVENT_AUDIT_COLUMNS)
        path = out / f"behavior_audit_{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    combined = out / "behavior_audit_summary.csv"
    pd.DataFrame(rows, columns=EVENT_AUDIT_COLUMNS).to_csv(combined, index=False)
    paths["summary"] = combined
    return paths


def build_audit_baseline_manifest(
    *,
    commit_hash: str,
    replay_scope: str,
    hyoas_manifest_path: str | Path,
    files: Sequence[str | Path],
) -> dict[str, object]:
    """Build a machine-readable frozen-baseline manifest with file hashes."""

    hyoas_manifest = json.loads(Path(hyoas_manifest_path).read_text(encoding="utf-8"))
    file_hashes = {
        str(Path(path)): sha256_file(path)
        for path in files
        if Path(path).exists() and Path(path).is_file()
    }
    return {
        "commit_hash": commit_hash,
        "replay_scope": replay_scope,
        "hyoas_source": hyoas_manifest.get("hyoas_source"),
        "audit_grade": hyoas_manifest.get("audit_grade"),
        "production_eligible": bool(hyoas_manifest.get("production_eligible", False)),
        "file_hashes": file_hashes,
    }


def freeze_replay_baseline(
    *,
    replay_dir: str | Path,
    audit_dir: str | Path,
    commit_hash: str,
) -> dict[str, object]:
    """Archive current replay CSVs, tail diagnostics, HYOAS manifest, and hashes."""

    replay_root = Path(replay_dir)
    out = Path(audit_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = [
        "weekly_replay.csv",
        "event_2008_09_to_2009_06.csv",
        "event_2020_02_to_2020_06.csv",
        "event_2021_10_to_2022_03.csv",
        "top_20_condition_number_reg.csv",
        "bottom_20_huber_weight.csv",
        "drift_flags.csv",
        "warmup_boundary_pm10.csv",
        "numerical_health_summary.json",
        "numerical_health_summary.md",
        "hyoas_archive_manifest.json",
    ]
    archived: list[Path] = []
    for name in names:
        src = replay_root / name
        if src.exists():
            dst = out / name
            shutil.copy2(src, dst)
            archived.append(dst)
    hyoas_path = out / "hyoas_archive_manifest.json"
    manifest = build_audit_baseline_manifest(
        commit_hash=commit_hash,
        replay_scope="state_stress_only",
        hyoas_manifest_path=hyoas_path,
        files=archived,
    )
    manifest_path = out / "audit_baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _window_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(frame["week_end"])
    return frame[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()


def _mean_abs_diff(left: pd.Series, right: pd.Series) -> float:
    joined = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")],
        axis=1,
    ).dropna()
    if joined.empty:
        return float("nan")
    return float((joined.iloc[:, 0] - joined.iloc[:, 1]).abs().mean())


def summarize_source_sensitivity(
    replays: Mapping[str, pd.DataFrame],
    windows: Mapping[str, tuple[str, str]] = EVENT_WINDOWS,
    *,
    reference_source: str,
) -> pd.DataFrame:
    """Compare HYOAS-source replay variants against a reference source."""

    if reference_source not in replays:
        raise ValueError(f"reference source missing: {reference_source}")
    rows: list[dict[str, object]] = []
    reference = replays[reference_source].copy()
    reference["week_end"] = pd.to_datetime(reference["week_end"])
    for source, replay in replays.items():
        current = replay.copy()
        current["week_end"] = pd.to_datetime(current["week_end"])
        for window, (start, end) in windows.items():
            left = _window_slice(current, start, end).set_index("week_end")
            right = _window_slice(reference, start, end).set_index("week_end")
            aligned = left.join(right, how="outer", lsuffix="", rsuffix="_reference")
            left_seq = _compressed_sequence(aligned.get("state_label", pd.Series(dtype=object)))
            right_seq = _compressed_sequence(
                aligned.get("state_label_reference", pd.Series(dtype=object))
            )
            left_drift = pd.to_numeric(aligned.get("drift_flag"), errors="coerce")
            right_drift = pd.to_numeric(
                aligned.get("drift_flag_reference"), errors="coerce"
            )
            drift_pairs = pd.concat([left_drift, right_drift], axis=1).dropna()
            rows.append(
                {
                    "source": source,
                    "reference_source": reference_source,
                    "window": window,
                    "rows_total": int(len(left)),
                    "rows_finite_H_t": int(
                        pd.to_numeric(left.get("H_t"), errors="coerce").notna().sum()
                    ),
                    "rows_finite_s_t": int(
                        pd.to_numeric(left.get("s_t"), errors="coerce").notna().sum()
                    ),
                    "H_t_mean_abs_diff_vs_reference": _mean_abs_diff(
                        aligned.get("H_t"), aligned.get("H_t_reference")
                    ),
                    "s_t_mean_abs_diff_vs_reference": _mean_abs_diff(
                        aligned.get("s_t"), aligned.get("s_t_reference")
                    ),
                    "drift_flag_mismatch_rows_vs_reference": int(
                        (drift_pairs.iloc[:, 0].astype(int) != drift_pairs.iloc[:, 1].astype(int)).sum()
                    )
                    if not drift_pairs.empty
                    else 0,
                    "window_state_sequence": left_seq,
                    "reference_window_state_sequence": right_seq,
                    "window_state_sequence_matches_reference": bool(left_seq == right_seq),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["window_state_sequence_matches_reference"] = frame[
            "window_state_sequence_matches_reference"
        ].astype(object)
    return frame


def build_replay_with_hyoas_source(
    weekly_inputs: pd.DataFrame,
    hyoas: pd.Series,
) -> pd.DataFrame:
    """Build a diagnostic replay after replacing only weekly HYOAS input."""

    inputs = weekly_inputs.copy()
    if "week_end" in inputs.columns:
        inputs = inputs.set_index(pd.to_datetime(inputs["week_end"])).drop(columns=["week_end"])
    else:
        inputs.index = pd.to_datetime(inputs.index)
    hyoas_weekly = hyoas.sort_index().resample("W-FRI").last()
    inputs["BAMLH0A0HYM2"] = hyoas_weekly.reindex(inputs.index)
    return build_replay_bundle(inputs).weekly


def write_source_sensitivity_report(summary: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    """Write CSV and Markdown source-sensitivity reports."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "source_sensitivity_summary.csv"
    md_path = out / "source_sensitivity_summary.md"
    summary.to_csv(csv_path, index=False)
    lines = [
        "# Source Sensitivity Summary",
        "",
        "Comparison fields: H_t, s_t, drift_flag, and window-level state sequence.",
        "",
        "| source | window | rows_finite_H_t | rows_finite_s_t | mean_abs_H_t | mean_abs_s_t | drift_mismatch_rows | sequence_match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        h = row["H_t_mean_abs_diff_vs_reference"]
        s = row["s_t_mean_abs_diff_vs_reference"]
        h_text = "" if pd.isna(h) else f"{h:.6g}"
        s_text = "" if pd.isna(s) else f"{s:.6g}"
        lines.append(
            f"| {row['source']} | {row['window']} | {row['rows_finite_H_t']} | "
            f"{row['rows_finite_s_t']} | {h_text} | {s_text} | "
            f"{row['drift_flag_mismatch_rows_vs_reference']} | "
            f"{row['window_state_sequence_matches_reference']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path
