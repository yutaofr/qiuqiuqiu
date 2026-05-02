"""Discord webhook sender for the weekly digest insight flow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import re
from typing import Any
from urllib import error, request


DESCRIPTION_LIMIT = 4096
DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_ATTEMPTS = 3
WEBHOOK_ENV_VARS = ("DISCORD_WEBHOOK_URL", "ALERT_WEBHOOK_URL")
USER_AGENT = "Mozilla/5.0"


@dataclass(frozen=True)
class PreparedPayload:
    payload: dict[str, Any]
    truncated: bool
    content_length: int
    preview: str


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _truncate_description(text: str, limit: int = DESCRIPTION_LIMIT) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    suffix = "\n\n[truncated]"
    if len(suffix) >= limit:
        return text[:limit], True
    return text[: limit - len(suffix)] + suffix, True


def _clean_latex(text: str) -> str:
    """Strip LaTeX delimiters like $h_t$ or \\(s_t\\) for Discord compatibility."""
    # 1. Handle common LaTeX parentheses
    text = re.sub(r"\\\((.*?)\\\)", r"\1", text)

    # 2. Strip paired dollar signs
    text = re.sub(r"\$([^$]+)\$", r"\1", text)

    # 3. Handle specific common commands (with or without backslashes)
    replacements = {
        r"\hat{k}_t": "k_hat_t",
        r"\hat{p}_t": "p_hat_t",
        r"\rho_t": "rho_t",
        r"\sigma_t": "sigma_t",
        r"\theta_t": "theta_t",
        r"\omega_t": "omega_t",
        r"\beta_t": "beta_t",
        r"\alpha_t": "alpha_t",
        r"\delta_t": "delta_t",
        r"\gamma_t": "gamma_t",
        "h_t": "h_t",
        "s_t": "s_t",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # 4. Final aggressive cleanup: strip ALL remaining $, \, {, }
    # We don't use $ for currency in this report (we use USD).
    for char in ["$", "\\", "{", "}"]:
        text = text.replace(char, "")

    return text


def _build_insight_payload(markdown: str, source_path: Path) -> PreparedPayload:
    cleaned = _clean_latex(markdown)
    description, truncated = _truncate_description(cleaned)
    embed: dict[str, Any] = {
        "title": "Weekly Digest Insight",
        "description": description,
    }
    if truncated:
        embed["footer"] = {"text": f"Local artifact for operator: {source_path}"}
    payload = {
        "allowed_mentions": {"parse": []},
        "embeds": [embed],
    }
    preview = description[:512]
    return PreparedPayload(
        payload=payload,
        truncated=truncated,
        content_length=len(description),
        preview=preview,
    )


def _validated_digest(validated_json: dict[str, Any], stage: str, message: str) -> str:
    phase14 = validated_json.get("phase14") or {}
    phase15 = validated_json.get("phase15") or {}
    digest = {
        "week_end": validated_json.get("week_end"),
        "phase14_mode": phase14.get("mode"),
        "phase14_backfill_mode": phase14.get("backfill_mode"),
        "phase15_signal_eligible": phase15.get("signal_eligible"),
        "phase15_execution_allowed": phase15.get("execution_allowed"),
        "phase15_orders_count": phase15.get("orders_count"),
        "phase15_reason": phase15.get("reason"),
        "error_stage": stage,
        "message": message,
    }
    return _canonical_json(digest)


def _build_fallback_payload(validated_json: dict[str, Any], stage: str, message: str, source_path: Path) -> PreparedPayload:
    digest = _validated_digest(validated_json, stage=stage, message=message)
    digest_hash = hashlib.sha256(digest.encode("utf-8")).hexdigest()
    description = digest
    embed: dict[str, Any] = {
        "title": message,
        "description": description,
        "footer": {
            "text": f"Validated artifact for operator: {source_path} | digest={digest_hash}",
        },
    }
    payload = {
        "allowed_mentions": {"parse": []},
        "embeds": [embed],
    }
    return PreparedPayload(
        payload=payload,
        truncated=False,
        content_length=len(description),
        preview=description,
    )


def _post_webhook(webhook_url: str, payload: dict[str, Any]) -> None:
    body = _canonical_json(payload).encode("utf-8")
    req = request.Request(
        webhook_url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "*/*",
            "User-Agent": USER_AGENT,
        },
    )
    for attempt in range(MAX_ATTEMPTS):
        try:
            with request.urlopen(req, timeout=DEFAULT_TIMEOUT_SECONDS) as resp:
                status = getattr(resp, "status", 200)
                if 200 <= status < 300:
                    return
                raise RuntimeError(f"Discord webhook returned non-2xx status {status}")
        except error.HTTPError as exc:
            if exc.code == 429 and attempt < MAX_ATTEMPTS - 1:
                retry_after_raw = exc.headers.get("Retry-After", "1") if exc.headers is not None else "1"
                try:
                    retry_after = float(retry_after_raw)
                except (TypeError, ValueError):
                    retry_after = 1.0
                time.sleep(retry_after)
                continue
            raise RuntimeError(f"Discord webhook failed with HTTP {exc.code}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Discord webhook request failed: {exc.reason}") from exc


def _resolve_webhook_url() -> str | None:
    for env_name in WEBHOOK_ENV_VARS:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def _print_dry_run(mode: str, prepared: PreparedPayload, source_path: Path, extra: dict[str, Any] | None = None) -> int:
    summary = {
        "mode": mode,
        "dry_run": True,
        "source_path": str(source_path),
        "truncated": prepared.truncated,
        "content_length": prepared.content_length,
        "preview": prepared.preview,
        "payload_keys": sorted(prepared.payload.keys()),
    }
    if extra:
        summary.update(extra)
    print(_canonical_json(summary))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("insight", "fallback_error"), required=True)
    parser.add_argument("--input", type=Path, help="Markdown input for insight mode.")
    parser.add_argument("--stage", help="Failure stage for fallback mode.")
    parser.add_argument("--validated-json", type=Path, help="Sanitized JSON input for fallback mode.")
    parser.add_argument("--message", help="Human-readable message for fallback mode.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.mode == "insight":
        if args.input is None:
            print("--input is required for insight mode", file=sys.stderr)
            return 2
        markdown = args.input.read_text(encoding="utf-8")
        prepared = _build_insight_payload(markdown, args.input)
        if args.dry_run:
            return _print_dry_run("insight", prepared, args.input)
        webhook_url = _resolve_webhook_url()
        if not webhook_url:
            print("DISCORD_WEBHOOK_URL or ALERT_WEBHOOK_URL is required unless --dry-run is set", file=sys.stderr)
            return 2
        try:
            _post_webhook(webhook_url, prepared.payload)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print("discord insight sent")
        return 0

    if args.validated_json is None or args.stage is None or args.message is None:
        print("--stage, --validated-json, and --message are required for fallback_error mode", file=sys.stderr)
        return 2
    validated = _load_json(args.validated_json)
    prepared = _build_fallback_payload(validated, stage=args.stage, message=args.message, source_path=args.validated_json)
    if args.dry_run:
        return _print_dry_run(
            "fallback_error",
            prepared,
            args.validated_json,
            extra={"stage": args.stage, "message": args.message},
        )
    webhook_url = _resolve_webhook_url()
    if not webhook_url:
        print("DISCORD_WEBHOOK_URL or ALERT_WEBHOOK_URL is required unless --dry-run is set", file=sys.stderr)
        return 2
    try:
        _post_webhook(webhook_url, prepared.payload)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"discord fallback sent for stage={args.stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
