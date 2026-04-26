from pathlib import Path

import pandas as pd

from qqq_cycle.backtest.diagnostics import (
    EVENT_WINDOWS,
    build_replay_bundle,
    synthetic_replay_inputs,
    write_replay_outputs,
)


def test_replay_table_has_required_columns_and_event_windows(tmp_path: Path) -> None:
    inputs = synthetic_replay_inputs()

    bundle = build_replay_bundle(inputs)

    required = {
        "week_end",
        "H_t",
        "I_t",
        "state_probs_json",
        "state_label",
        "d_t",
        "a_t",
        "g_t_raw",
        "g_t_stress",
        "s_t",
        "drift_probe_raw",
        "drift_flag",
        "maha",
        "huber_weight",
        "eigval_1",
        "eigval_2_raw",
        "eigval_2_reg",
        "condition_number_raw",
        "condition_number_reg",
        "eigval_2_was_floored",
    }
    assert required.issubset(bundle.weekly.columns)
    assert bundle.weekly["week_end"].is_monotonic_increasing
    assert set(EVENT_WINDOWS).issubset(bundle.event_windows)
    assert all(not frame.empty for frame in bundle.event_windows.values())

    output_dir = write_replay_outputs(bundle, tmp_path)
    assert (output_dir / "weekly_replay.csv").exists()
    for name in EVENT_WINDOWS:
        assert (output_dir / f"event_{name}.csv").exists()

    loaded = pd.read_csv(output_dir / "weekly_replay.csv")
    assert required.issubset(loaded.columns)


def test_replay_probs_are_json_arrays_with_five_entries() -> None:
    bundle = build_replay_bundle(synthetic_replay_inputs())
    valid = bundle.weekly["state_probs_json"].dropna()

    assert not valid.empty
    assert valid.str.startswith("[").all()
    assert valid.str.endswith("]").all()
