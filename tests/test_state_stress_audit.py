import json
from pathlib import Path

import numpy as np
import pandas as pd

from qqq_cycle.backtest.diagnostics import EVENT_WINDOWS
from qqq_cycle.backtest.state_stress_audit import (
    build_audit_baseline_manifest,
    summarize_behavior_window,
    summarize_source_sensitivity,
)


def _weekly_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-02-07", periods=8, freq="W-FRI")
    return pd.DataFrame(
        {
            "week_end": dates.strftime("%Y-%m-%d"),
            "H_t": [1.0, 0.8, -0.3, -0.8, -1.1, -0.7, -0.2, 0.1],
            "s_t": [0.1, 0.2, 0.3, 0.7, 0.95, 0.8, 0.4, 0.2],
            "state_label": ["S5", "S5", "S3", "S1", "S1", "S2", "S3", "S4"],
            "drift_flag": [0, 0, 0, 1, 0, 0, 0, 0],
        }
    )


def test_behavior_window_summary_reports_required_state_stress_fields() -> None:
    weekly = _weekly_frame()
    summary = summarize_behavior_window(
        weekly,
        window_name="2020_02_to_2020_06",
        start="2020-02-01",
        end="2020-06-30",
        stress_upper_tail_threshold=0.75,
    )

    assert summary["rows_total"] == 8
    assert summary["rows_finite_H_t"] == 8
    assert summary["rows_finite_s_t"] == 8
    assert summary["dominant_state_sequence"] == "S5 -> S3 -> S1 -> S2 -> S3 -> S4"
    assert summary["first_week_low_heat_states_become_material"] == "2020-02-28"
    assert summary["first_week_s_t_breaks_into_upper_tail_regime"] == "2020-03-06"
    assert summary["drift_flag_rows"] == 1
    assert summary["lag_weeks_between_state_migration_and_stress_breakout"] == 1


def test_audit_manifest_hashes_files_and_preserves_scope(tmp_path: Path) -> None:
    replay = tmp_path / "weekly_replay.csv"
    replay.write_text("week_end,H_t\n2020-01-03,1.0\n", encoding="utf-8")
    hyoas_manifest = tmp_path / "hyoas_archive_manifest.json"
    hyoas_manifest.write_text(
        json.dumps(
            {
                "hyoas_source": "csv_override",
                "audit_grade": "conditional",
                "production_eligible": False,
            }
        ),
        encoding="utf-8",
    )

    manifest = build_audit_baseline_manifest(
        commit_hash="abc123",
        replay_scope="state_stress_only",
        hyoas_manifest_path=hyoas_manifest,
        files=[replay, hyoas_manifest],
    )

    assert manifest["commit_hash"] == "abc123"
    assert manifest["replay_scope"] == "state_stress_only"
    assert manifest["hyoas_source"] == "csv_override"
    assert manifest["audit_grade"] == "conditional"
    assert manifest["production_eligible"] is False
    assert set(manifest["file_hashes"]) == {str(replay), str(hyoas_manifest)}


def test_source_sensitivity_summary_compares_metrics_and_state_sequences() -> None:
    base = _weekly_frame()
    shifted = base.copy()
    shifted["H_t"] = shifted["H_t"] + 0.25
    shifted["s_t"] = shifted["s_t"] + 0.1
    shifted.loc[3, "state_label"] = "S2"
    shifted.loc[3, "drift_flag"] = 0
    shifted.loc[4, "drift_flag"] = 1

    summary = summarize_source_sensitivity(
        {
            "eco_archive_only": base,
            "eco_archive_plus_equibles": shifted,
        },
        EVENT_WINDOWS,
        reference_source="eco_archive_plus_equibles",
    )

    row = summary[
        (summary["source"] == "eco_archive_only")
        & (summary["window"] == "2020_02_to_2020_06")
    ].iloc[0]
    assert row["reference_source"] == "eco_archive_plus_equibles"
    assert row["window"] == "2020_02_to_2020_06"
    assert np.isclose(row["H_t_mean_abs_diff_vs_reference"], 0.25)
    assert np.isclose(row["s_t_mean_abs_diff_vs_reference"], 0.1)
    assert row["drift_flag_mismatch_rows_vs_reference"] == 2
    assert row["window_state_sequence_matches_reference"] is False
