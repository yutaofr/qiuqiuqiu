from __future__ import annotations

import json
from pathlib import Path

from qqq_cycle.portfolio.signal_gate import evaluate_signal_eligibility


def _strict_snapshot(tmp_path: Path) -> dict:
    path = tmp_path / "synthetic_phase14_snapshot.json"
    snapshot = {
        "week_end": "2026-04-24",
        "mode": "strict",
        "backfill_mode": "strict_recovery",
        "strict_gate_passed": True,
        "execution_permitted": True,
        "h_t": 0.25,
        "rho_t": 0.34,
        "k_hat_t": 2,
        "s_t": 0.11,
    }
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def test_real_degraded_backfill_snapshot_is_not_eligible() -> None:
    snapshot = {
        "week_end": "2026-04-24",
        "mode": "strict",
        "backfill_mode": "degraded_backfill",
        "strict_gate_passed": True,
        "execution_permitted": True,
        "h_t": 0.2,
        "rho_t": 0.3,
        "k_hat_t": 1,
        "s_t": 0.1,
    }

    result = evaluate_signal_eligibility(
        snapshot,
        paper_only=True,
        broker_submission_allowed=False,
    )

    assert result.signal_eligible is False
    assert result.execution_allowed is False
    assert result.reason == "degraded_backfill_signal"


def test_block_snapshot_is_not_eligible() -> None:
    result = evaluate_signal_eligibility(
        {"week_end": "2026-04-24", "mode": "strict", "backfill_mode": "block"},
        paper_only=True,
        broker_submission_allowed=False,
    )

    assert result.signal_eligible is False
    assert result.reason == "block_signal"


def test_strict_snapshot_in_tmp_path_is_eligible(tmp_path: Path) -> None:
    result = evaluate_signal_eligibility(
        _strict_snapshot(tmp_path),
        paper_only=True,
        broker_submission_allowed=False,
    )

    assert result.signal_eligible is True
    assert result.execution_allowed is True
    assert result.reason == "eligible_strict_signal"


def test_rho_missing_is_not_eligible(tmp_path: Path) -> None:
    snapshot = _strict_snapshot(tmp_path)
    snapshot["rho_t"] = None

    result = evaluate_signal_eligibility(
        snapshot,
        paper_only=True,
        broker_submission_allowed=False,
    )

    assert result.signal_eligible is False
    assert result.reason == "rho_t_missing"


def test_strict_gate_failed_is_not_eligible(tmp_path: Path) -> None:
    snapshot = _strict_snapshot(tmp_path)
    snapshot["strict_gate_passed"] = False

    result = evaluate_signal_eligibility(
        snapshot,
        paper_only=True,
        broker_submission_allowed=False,
    )

    assert result.signal_eligible is False
    assert result.reason == "strict_gate_failed"


def test_execution_not_permitted_is_not_eligible(tmp_path: Path) -> None:
    snapshot = _strict_snapshot(tmp_path)
    snapshot["execution_permitted"] = False

    result = evaluate_signal_eligibility(
        snapshot,
        paper_only=True,
        broker_submission_allowed=False,
    )

    assert result.signal_eligible is False
    assert result.reason == "execution_not_permitted"


def test_paper_only_false_is_rejected(tmp_path: Path) -> None:
    result = evaluate_signal_eligibility(
        _strict_snapshot(tmp_path),
        paper_only=False,
        broker_submission_allowed=False,
    )

    assert result.signal_eligible is False
    assert result.reason == "paper_only_invariant_failed"


def test_broker_submission_allowed_true_is_rejected(tmp_path: Path) -> None:
    result = evaluate_signal_eligibility(
        _strict_snapshot(tmp_path),
        paper_only=True,
        broker_submission_allowed=True,
    )

    assert result.signal_eligible is False
    assert result.reason == "paper_only_invariant_failed"
