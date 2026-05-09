from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from src.output import send_insight


def _write_markdown(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _write_validated_json(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "week_end": "2026-04-24",
                "generated_at_utc": "2026-04-29T10:00:00Z",
                "system": "qiuqiuqiu",
                "phase14": {
                    "mode": "degraded",
                    "backfill_mode": "degraded_backfill",
                    "strict_gate_passed": False,
                    "micro_state_frozen": True,
                    "h_t": None,
                    "rho_t": None,
                    "k_hat_t": None,
                    "s_t": None,
                },
                "phase15": {
                    "paper_only": True,
                    "broker_submission_allowed": False,
                    "signal_eligible": False,
                    "execution_allowed": False,
                    "orders_count": 0,
                    "reason": "degraded_backfill_signal",
                },
                "sanitization": {
                    "sanitized": True,
                    "removed_fields_count": 4,
                    "policy": "weekly_digest_allowlist_v1",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_dry_run_does_not_call_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    markdown_path = _write_markdown(tmp_path / "insight.md", "# Weekly Digest\n\nhello")
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

    called = {"value": False}

    def _fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        called["value"] = True
        raise AssertionError("HTTP should not be called in dry-run")

    monkeypatch.setattr(send_insight.request, "urlopen", _fail)

    exit_code = send_insight.main(
        [
            "--mode",
            "insight",
            "--input",
            str(markdown_path),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert called["value"] is False
    assert payload["dry_run"] is True
    assert payload["mode"] == "insight"


def test_missing_webhook_fails_outside_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    markdown_path = _write_markdown(tmp_path / "insight.md", "# Weekly Digest\n\nhello")
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)

    exit_code = send_insight.main(
        [
            "--mode",
            "insight",
            "--input",
            str(markdown_path),
        ]
    )

    assert exit_code != 0


def test_alert_webhook_url_is_accepted_outside_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    markdown_path = _write_markdown(tmp_path / "insight.md", "# Weekly Digest\n\nhello")
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://discord.example/alert-webhook")

    called = {"value": False}

    def _fake_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
        called["value"] = True
        assert req.get_header("User-agent") == "Mozilla/5.0"

        class _Resp:
            status = 204

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Resp()

    monkeypatch.setattr(send_insight.request, "urlopen", _fake_urlopen)

    exit_code = send_insight.main(
        [
            "--mode",
            "insight",
            "--input",
            str(markdown_path),
        ]
    )

    assert exit_code == 0
    assert called["value"] is True


def test_long_markdown_is_truncated_in_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    markdown_path = _write_markdown(tmp_path / "insight.md", "A" * 5000)

    exit_code = send_insight.main(
        [
            "--mode",
            "insight",
            "--input",
            str(markdown_path),
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["truncated"] is True
    assert payload["content_length"] <= 4096


def test_fallback_digest_is_deterministic_and_not_full_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    validated_path = _write_validated_json(tmp_path / "validated.json")
    exit_code = send_insight.main(
        [
            "--mode",
            "fallback_error",
            "--stage",
            "gemini",
            "--validated-json",
            str(validated_path),
            "--message",
            "[ERROR] AI Interpretation Failed.",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["mode"] == "fallback_error"
    assert payload["message"] == "[ERROR] AI Interpretation Failed."
    assert "degraded_backfill_signal" in payload["preview"]
    assert json.dumps(json.loads(validated_path.read_text(encoding="utf-8"))) not in payload["preview"]


def test_retry_after_429_is_honored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    markdown_path = _write_markdown(tmp_path / "insight.md", "# Weekly Digest\n\nhello")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook")

    calls: list[Request] = []

    class _Resp:
        status = 204

        def read(self) -> bytes:
            return b""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_urlopen(req: Request, timeout: float | None = None):  # type: ignore[no-untyped-def]
        calls.append(req)
        if len(calls) == 1:
            raise HTTPError(req.full_url, 429, "Too Many Requests", {"Retry-After": "0"}, None)
        return _Resp()

    sleeps: list[float] = []

    monkeypatch.setattr(send_insight.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(send_insight.time, "sleep", lambda seconds: sleeps.append(seconds))

    exit_code = send_insight.main(
        [
            "--mode",
            "insight",
            "--input",
            str(markdown_path),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 2
    assert sleeps == [0.0]


def test_webhook_is_not_logged_in_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    markdown_path = _write_markdown(tmp_path / "insight.md", "# Weekly Digest\n\nhello")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook-secret")

    exit_code = send_insight.main(
        [
            "--mode",
            "insight",
            "--input",
            str(markdown_path),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "webhook-secret" not in captured.out
    assert "discord.example" not in captured.out


def test_retry_after_too_large_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    markdown_path = _write_markdown(tmp_path / "insight.md", "# Weekly Digest\n\nhello")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook")

    def _fake_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
        raise HTTPError(req.full_url, 429, "Too Many Requests", {"Retry-After": "600"}, None)

    monkeypatch.setattr(send_insight.request, "urlopen", _fake_urlopen)

    exit_code = send_insight.main(
        [
            "--mode",
            "insight",
            "--input",
            str(markdown_path),
        ]
    )

    assert exit_code != 0


def test_retry_after_logs_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    markdown_path = _write_markdown(tmp_path / "insight.md", "# Weekly Digest\n\nhello")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook")

    calls = []

    def _fake_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
        calls.append(req)
        if len(calls) == 1:
            raise HTTPError(req.full_url, 429, "Too Many Requests", {"Retry-After": "0.1"}, None)

        class _Resp:
            status = 204

            def __enter__(self):
                return self

            def __exit__(self, *args):  # type: ignore[no-untyped-def]
                pass

        return _Resp()

    monkeypatch.setattr(send_insight.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(send_insight.time, "sleep", lambda x: None)

    exit_code = send_insight.main(
        [
            "--mode",
            "insight",
            "--input",
            str(markdown_path),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Discord 429: Rate limited. Retrying after 0.1s" in captured.err
