from pathlib import Path
import json

from qqq_cycle.backtest.diagnostics import build_replay_bundle, synthetic_replay_inputs
from qqq_cycle.backtest.oos_eval import summarize_numerical_health, write_health_summary


def test_numerical_health_summary_contains_required_metrics(tmp_path: Path) -> None:
    replay = build_replay_bundle(synthetic_replay_inputs()).weekly

    summary = summarize_numerical_health(replay)

    for metric in [
        "maha",
        "huber_weight",
        "condition_number_raw",
        "condition_number_reg",
    ]:
        assert metric in summary["distributions"]
        assert "p50" in summary["distributions"][metric]
    assert "eigval_2_was_floored_frequency" in summary["frequencies"]
    assert "state_health_degradation_frequency" in summary["frequencies"]
    assert "warmup_rows" in summary["coverage"]
    assert "drift_flag_count" in summary["counts"]

    json_path, md_path = write_health_summary(summary, tmp_path)
    assert json_path.exists()
    assert md_path.exists()
    loaded = json.loads(json_path.read_text())
    assert loaded["counts"]["rows"] == summary["counts"]["rows"]
    assert "Numerical Health Summary" in md_path.read_text()
