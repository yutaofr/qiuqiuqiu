import json
from pathlib import Path

import numpy as np
import pandas as pd

from qqq_cycle.backtest.diagnostics import EVENT_WINDOWS
from qqq_cycle.backtest.state_stress_audit import (
    build_audit_baseline_manifest,
    build_warmup_dependency_map,
    explain_warmup_boundary,
    summarize_behavior_window,
    summarize_source_sensitivity,
    write_warmup_explanation,
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
    assert summary["state_label_status"] == "valid"


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


def test_warmup_dependency_map_identifies_covariance_binding_for_2008() -> None:
    idx = pd.date_range("2000-01-07", periods=620, freq="W-FRI")
    n = len(idx)
    weekly_inputs = pd.DataFrame(
        {
            "week_end": idx,
            "DFII10": np.linspace(0.1, 0.5, n),
            "DGS2": np.linspace(2.0, 3.0, n),
            "BAMLH0A0HYM2": np.linspace(4.0, 5.0, n),
            "NFCI": np.linspace(-0.4, 0.2, n),
            "VIXCLS": np.linspace(15.0, 25.0, n),
            "USEPUINDXD": np.linspace(50.0, 90.0, n),
            "AI_GPR": np.linspace(20.0, 70.0, n),
            "QQQ": 100.0 * np.exp(np.linspace(0.0, 1.0, n)),
        }
    )
    replay = pd.DataFrame(
        {
            "week_end": idx,
            "state_label": ["WARMUP"] * 520 + ["S3"] * 100,
            "is_warm": [False] * 519 + [True] * 101,
        }
    )

    warmup_map = build_warmup_dependency_map(weekly_inputs, replay)
    explanation = explain_warmup_boundary(
        warmup_map,
        window_name="2008_09_to_2009_06",
        start="2008-09-01",
        end="2009-06-30",
    )

    assert explanation["is_blocked"] is True
    assert explanation["binding_stage"] == "covariance_warmup"
    assert explanation["state_label_earliest_valid_date"] == idx[520].strftime("%Y-%m-%d")
    cov_row = warmup_map[warmup_map["stage"] == "covariance_warmup"].iloc[0]
    assert explanation["first_finite_theta_date"] == cov_row["first_finite_transformed_date"]
    assert "260 finite Theta" in cov_row["blocking_reason"]


def test_warmup_explanation_artifacts_match_pipeline_availability(tmp_path: Path) -> None:
    warmup_map = pd.DataFrame(
        [
            {
                "stage": "state_label",
                "first_available_raw_date": None,
                "first_finite_transformed_date": None,
                "first_finite_weekly_date": None,
                "first_date_usable_for_state_probability": "2010-01-22",
                "first_date_usable_for_stress": None,
                "blocking_reason": "locked",
            }
        ]
    )
    explanation = {
        "window": "2008_09_to_2009_06",
        "start": "2008-09-01",
        "end": "2009-06-30",
        "is_blocked": True,
        "binding_stage": "covariance_warmup",
        "first_finite_theta_date": "2005-01-28",
        "covariance_earliest_usable_date": "2010-01-15",
        "state_label_earliest_valid_date": "2010-01-22",
        "dependency_chain": ["Theta(H_t,I_t)", "260 finite-Theta covariance warmup"],
        "blocking_reason": "window ends before state labels unlock",
    }

    _, json_path, md_path = write_warmup_explanation(warmup_map, explanation, tmp_path)

    payload = json.loads(json_path.read_text())
    assert payload["binding_stage"] == "covariance_warmup"
    assert payload["state_label_earliest_valid_date"] == "2010-01-22"
    assert "2010-01-22" in md_path.read_text()


def test_behavior_summary_marks_formally_blocked_window() -> None:
    weekly = _weekly_frame()
    explanation = {
        "is_blocked": True,
        "binding_stage": "covariance_warmup",
        "state_label_earliest_valid_date": "2010-01-22",
        "blocking_reason": "blocked by 260 finite-Theta warmup",
    }

    summary = summarize_behavior_window(
        weekly,
        window_name="2008_09_to_2009_06",
        start="2020-02-01",
        end="2020-06-30",
        stress_upper_tail_threshold=0.75,
        warmup_explanation=explanation,
    )

    assert summary["state_label_status"] == "mathematically_blocked"
    assert summary["state_label_blocking_stage"] == "covariance_warmup"
    assert summary["state_label_earliest_valid_date"] == "2010-01-22"


def test_warmup_audit_does_not_expand_scope_columns() -> None:
    weekly = _weekly_frame()
    forbidden = {"h_t", "rho_t", "return", "returns", "micro"}

    summary = summarize_behavior_window(
        weekly,
        window_name="2020_02_to_2020_06",
        start="2020-02-01",
        end="2020-06-30",
        stress_upper_tail_threshold=0.75,
    )

    assert forbidden.isdisjoint(set(summary))
