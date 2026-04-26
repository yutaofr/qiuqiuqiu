import pandas as pd
from pathlib import Path

from qqq_cycle.backtest.diagnostics import build_replay_bundle, synthetic_replay_inputs
from qqq_cycle.backtest.oos_eval import build_tail_diagnostics, write_tail_diagnostics


def test_tail_diagnostics_contains_required_extracts() -> None:
    replay = build_replay_bundle(synthetic_replay_inputs()).weekly

    tails = build_tail_diagnostics(replay)

    assert set(tails) == {
        "top_20_condition_number_reg",
        "bottom_20_huber_weight",
        "drift_flags",
        "warmup_boundary_pm10",
    }
    assert len(tails["top_20_condition_number_reg"]) <= 20
    assert len(tails["bottom_20_huber_weight"]) <= 20
    assert (pd.to_numeric(tails["drift_flags"]["drift_flag"]) == 1).all()
    assert not tails["warmup_boundary_pm10"].empty


def test_write_tail_diagnostics_emits_required_files(tmp_path: Path) -> None:
    replay = build_replay_bundle(synthetic_replay_inputs()).weekly

    paths = write_tail_diagnostics(replay, tmp_path)

    assert set(paths) == {
        "top_20_condition_number_reg",
        "bottom_20_huber_weight",
        "drift_flags",
        "warmup_boundary_pm10",
    }
    assert all(path.exists() for path in paths.values())
