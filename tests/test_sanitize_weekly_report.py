from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sanitizer_allowlist_removes_sensitive_and_nonportable_fields(tmp_path: Path) -> None:
    report_path = tmp_path / "weekly_report.json"
    sanitized_path = tmp_path / "weekly_report_sanitized.json"
    report_path.write_text(
        json.dumps(
            {
                "week_end": "2026-04-24",
                "generated_at_utc": "2026-04-29T10:00:00Z",
                "system": "qiuqiuqiu",
                "source": "weekly_digest",
                "phase14": {
                    "snapshot_path": "/Users/tester/projects/qiuqiuqiu/outputs/phase14/cycle_snapshot_latest.json",
                    "mode": "degraded",
                    "backfill_mode": "degraded_backfill",
                    "strict_gate_passed": False,
                    "micro_state_frozen": True,
                    "h_t": None,
                    "rho_t": None,
                    "k_hat_t": None,
                    "s_t": None,
                    "api_key": "sk-test-123",
                    "webhook_url": "https://discord.com/api/webhooks/fake",
                },
                "phase15": {
                    "summary_path": "/Users/tester/projects/qiuqiuqiu/outputs/phase15/execution_sandbox_summary_latest.json",
                    "paper_only": True,
                    "broker_submission_allowed": False,
                    "signal_eligible": False,
                    "execution_allowed": False,
                    "orders_count": 0,
                    "reason": "degraded_backfill_signal",
                    "password": "secret",
                    "account_id": "acct-001",
                },
                "extra": {"user_email": "tester@example.com"},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "sanitize_weekly_report.py"),
            "--input",
            str(report_path),
            "--output",
            str(sanitized_path),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    sanitized = json.loads(sanitized_path.read_text(encoding="utf-8"))
    text = sanitized_path.read_text(encoding="utf-8")
    assert sanitized["sanitization"]["sanitized"] is True
    assert sanitized["sanitization"]["policy"] == "weekly_digest_allowlist_v1"
    assert sanitized["sanitization"]["removed_fields_count"] >= 1
    assert sanitized["phase14"]["mode"] == "degraded"
    assert sanitized["phase15"]["paper_only"] is True
    assert "snapshot_path" not in sanitized["phase14"]
    assert "summary_path" not in sanitized["phase15"]
    assert "/Users/" not in text
    assert "api_key" not in text
    assert "webhook" not in text
    assert "tester@example.com" not in text


def test_sanitizer_output_is_valid_json(tmp_path: Path) -> None:
    report_path = tmp_path / "weekly_report.json"
    sanitized_path = tmp_path / "weekly_report_sanitized.json"
    report_path.write_text(
        json.dumps(
            {
                "week_end": "2026-04-24",
                "generated_at_utc": "2026-04-29T10:00:00Z",
                "system": "qiuqiuqiu",
                "phase14": {
                    "mode": "strict",
                    "backfill_mode": "strict_recovery",
                    "strict_gate_passed": True,
                    "micro_state_frozen": True,
                    "h_t": 0.1,
                    "rho_t": 0.2,
                    "k_hat_t": 2,
                    "s_t": 0.3,
                },
                "phase15": {
                    "paper_only": True,
                    "broker_submission_allowed": False,
                    "signal_eligible": True,
                    "execution_allowed": True,
                    "orders_count": 0,
                    "reason": "eligible_strict_signal",
                },
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "sanitize_weekly_report.py"),
            "--input",
            str(report_path),
            "--output",
            str(sanitized_path),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    json.loads(sanitized_path.read_text(encoding="utf-8"))
