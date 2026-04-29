from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_phase14_snapshot(base: Path, week_end: str = "2026-04-24") -> Path:
    path = base / "outputs" / "phase14" / "cycle_snapshot_latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "week_end": week_end,
                "mode": "degraded",
                "backfill_mode": None,
                "strict_gate_passed": False,
                "micro_state_frozen": True,
                "h_t": None,
                "rho_t": None,
                "k_hat_t": None,
                "s_t": None,
                "source_hash": "phase14hash",
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_phase15_summary(base: Path, week_end: str = "2026-04-24") -> Path:
    path = base / "outputs" / "phase15" / "execution_sandbox_summary_latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "week_end": week_end,
                "paper_only": True,
                "broker_submission_allowed": False,
                "signal_eligible": False,
                "execution_allowed": False,
                "orders_count": 0,
                "reason": "degraded_backfill_signal",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_main_writes_valid_json_and_keeps_stdout_clean(tmp_path: Path) -> None:
    _write_phase14_snapshot(tmp_path)
    _write_phase15_summary(tmp_path)
    output_path = tmp_path / "weekly_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src" / "main.py"),
            "--week-end",
            "2026-04-24",
            "--output",
            str(output_path),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["week_end"] == "2026-04-24"
    assert payload["phase15"]["paper_only"] is True
    assert payload["phase15"]["broker_submission_allowed"] is False
    assert payload["phase14"]["snapshot_path"].endswith("outputs/phase14/cycle_snapshot_latest.json")
    assert "{" not in result.stdout
    assert "phase15" not in result.stdout


def test_main_missing_phase15_summary_fails_cleanly(tmp_path: Path) -> None:
    _write_phase14_snapshot(tmp_path)
    output_path = tmp_path / "weekly_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src" / "main.py"),
            "--week-end",
            "2026-04-24",
            "--output",
            str(output_path),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert not output_path.exists()
    assert "scripts/run_phase15_sandbox.py" in result.stderr
