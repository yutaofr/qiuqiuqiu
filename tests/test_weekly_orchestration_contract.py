from __future__ import annotations

import json
import os
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
                "backfill_mode": "degraded_backfill",
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


def _write_stub_command(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_env_file(base: Path, *, mode: int, webhook_url: str = "https://discord.example/webhook") -> Path:
    path = base / ".env"
    path.write_text(f"DISCORD_WEBHOOK_URL={webhook_url}\n", encoding="utf-8")
    path.chmod(mode)
    return path


def test_dry_run_does_not_call_gemini_or_discord(tmp_path: Path) -> None:
    _write_phase14_snapshot(tmp_path)
    _write_phase15_summary(tmp_path)
    gemini_marker = tmp_path / "gemini_called"
    discord_marker = tmp_path / "discord_called"
    gemini_stub = _write_stub_command(
        tmp_path / "gemini_stub.sh",
        f"#!/bin/bash\nset -euo pipefail\ntouch {gemini_marker}\nexit 1\n",
    )
    discord_stub = _write_stub_command(
        tmp_path / "discord_stub.sh",
        f"#!/bin/bash\nset -euo pipefail\ntouch {discord_marker}\nexit 1\n",
    )

    env = os.environ.copy()
    env.update(
        {
            "GEMINI_CMD": str(gemini_stub),
            "DISCORD_WEBHOOK_URL": "https://discord.example/webhook",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_weekly_orchestration.sh"),
            "--dry-run",
            "--work-root",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert not gemini_marker.exists()
    assert not discord_marker.exists()
    status_path = tmp_path / ".temp" / "weekly" / "2026-04-24" / "run_status.json"
    assert status_path.exists()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["dry_run"] is True
    assert status["success"] is True


def test_orchestrator_computes_week_end_when_omitted_with_fake_clock(tmp_path: Path) -> None:
    _write_phase14_snapshot(tmp_path, week_end="2026-04-24")
    _write_phase15_summary(tmp_path, week_end="2026-04-24")

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_weekly_orchestration.sh"),
            "--dry-run",
            "--work-root",
            str(tmp_path),
            "--now-utc",
            "2026-04-24T21:00:00Z",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    status_path = tmp_path / ".temp" / "weekly" / "2026-04-24" / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["week_end"] == "2026-04-24"


def test_resend_requires_reason(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_weekly_orchestration.sh"),
            "--week-end",
            "2026-04-24",
            "--resend",
            "--work-root",
            str(tmp_path),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "resend-reason" in result.stderr.lower()


def test_run_lock_blocks_concurrent_execution(tmp_path: Path) -> None:
    _write_phase14_snapshot(tmp_path)
    _write_phase15_summary(tmp_path)
    lock_dir = tmp_path / ".temp" / "weekly" / "2026-04-24" / ".run_lock"
    lock_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_weekly_orchestration.sh"),
            "--week-end",
            "2026-04-24",
            "--dry-run",
            "--work-root",
            str(tmp_path),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "run_lock" in result.stderr


def test_stale_lock_requires_explicit_recovery_flag(tmp_path: Path) -> None:
    _write_phase14_snapshot(tmp_path)
    _write_phase15_summary(tmp_path)
    lock_dir = tmp_path / ".temp" / "weekly" / "2026-04-24" / ".run_lock"
    lock_dir.mkdir(parents=True, exist_ok=True)

    blocked = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_weekly_orchestration.sh"),
            "--week-end",
            "2026-04-24",
            "--dry-run",
            "--work-root",
            str(tmp_path),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert blocked.returncode != 0

    recovered = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_weekly_orchestration.sh"),
            "--week-end",
            "2026-04-24",
            "--dry-run",
            "--recover-stale-lock",
            "--work-root",
            str(tmp_path),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert recovered.returncode == 0, recovered.stderr


def test_sent_marker_prevents_duplicate_send(tmp_path: Path) -> None:
    _write_phase14_snapshot(tmp_path)
    _write_phase15_summary(tmp_path)
    week_dir = tmp_path / "outputs" / "weekly" / "2026-04-24"
    week_dir.mkdir(parents=True, exist_ok=True)
    (week_dir / "sent_discord.ok").write_text("ok\n", encoding="utf-8")

    gemini_marker = tmp_path / "gemini_called"
    gemini_stub = _write_stub_command(
        tmp_path / "gemini_stub.sh",
        f"#!/bin/bash\nset -euo pipefail\ntouch {gemini_marker}\nexit 1\n",
    )

    env = os.environ.copy()
    env["GEMINI_CMD"] = str(gemini_stub)
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_weekly_orchestration.sh"),
            "--week-end",
            "2026-04-24",
            "--work-root",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert not gemini_marker.exists()


def test_gemini_failure_emits_fallback_once(tmp_path: Path) -> None:
    _write_phase14_snapshot(tmp_path)
    _write_phase15_summary(tmp_path)
    weekly_root = tmp_path / "outputs" / "weekly"
    week_dir = weekly_root / "2026-04-24"
    week_dir.mkdir(parents=True, exist_ok=True)

    gemini_stub = _write_stub_command(
        tmp_path / "gemini_stub.sh",
        "#!/bin/bash\nset -euo pipefail\nexit 1\n",
    )
    webhook_server = tmp_path / "discord_server.py"
    request_count = tmp_path / "request_count.txt"
    webhook_server.write_text(
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "from pathlib import Path\n"
        "import sys\n"
        f"count_path = Path({str(request_count)!r})\n"
        "count_path.write_text('0', encoding='utf-8')\n"
        "class H(BaseHTTPRequestHandler):\n"
        "    def do_POST(self):\n"
        "        current = int(count_path.read_text(encoding='utf-8')) + 1\n"
        "        count_path.write_text(str(current), encoding='utf-8')\n"
        "        self.send_response(204)\n"
        "        self.end_headers()\n"
        "    def log_message(self, *args):\n"
        "        pass\n"
        "server = HTTPServer(('127.0.0.1', 0), H)\n"
        "print(server.server_port)\n"
        "sys.stdout.flush()\n"
        "for _ in range(2):\n"
        "    server.handle_request()\n"
    ,
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, str(webhook_server)],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    port = int(proc.stdout.readline().strip())

    env = os.environ.copy()
    env.update(
        {
            "GEMINI_CMD": str(gemini_stub),
            "DISCORD_WEBHOOK_URL": f"http://127.0.0.1:{port}/webhook",
        }
    )

    first = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_weekly_orchestration.sh"),
            "--week-end",
            "2026-04-24",
            "--work-root",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_weekly_orchestration.sh"),
            "--week-end",
            "2026-04-24",
            "--work-root",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    proc.terminate()
    proc.wait(timeout=10)

    assert first.returncode != 0
    assert second.returncode != 0
    assert (week_dir / "notified_error_gemini.ok").exists()
    assert request_count.read_text(encoding="utf-8") == "1"


def test_env_file_with_insecure_permissions_is_rejected(tmp_path: Path) -> None:
    _write_env_file(tmp_path, mode=0o644)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_weekly_orchestration.sh"),
            "--week-end",
            "2026-04-24",
            "--dry-run",
            "--work-root",
            str(tmp_path),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "0600" in result.stderr


def test_env_file_600_is_accepted(tmp_path: Path) -> None:
    _write_phase14_snapshot(tmp_path)
    _write_phase15_summary(tmp_path)
    _write_env_file(tmp_path, mode=0o600)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_weekly_orchestration.sh"),
            "--week-end",
            "2026-04-24",
            "--dry-run",
            "--work-root",
            str(tmp_path),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / ".temp" / "weekly" / "2026-04-24" / "run_status.json").exists()


def test_launchd_template_has_no_week_end_placeholder() -> None:
    template = (ROOT / "launchd" / "com.qiuqiuqiu.weekly.plist.template").read_text(encoding="utf-8")
    assert "__WEEK_END_PLACEHOLDER__" not in template
    assert "<string>--week-end</string>" not in template
    assert "<string>/bin/bash</string>" in template
    assert "<string>/ABSOLUTE/PATH/TO/scripts/run_weekly_orchestration.sh</string>" in template
