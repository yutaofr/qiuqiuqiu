#!/usr/bin/env python3
"""Allowlist sanitize the weekly digest report before it reaches Gemini."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any


POLICY = "weekly_digest_allowlist_v1"
ABSOLUTE_PATH_RE = re.compile(r"^(?:/Users/|/home/|/Volumes/|/private/|/var/|/tmp/|/[A-Za-z0-9_.-]+)")
SECRET_VALUE_RE = re.compile(
    r"(https?://[^\\s\"']*discord(?:app)?\\.com/api/webhooks/|\\bsk-[A-Za-z0-9-]{8,}\\b|\\b[A-Za-z0-9_-]{24,}\\.[A-Za-z0-9_-]{6,}\\.[A-Za-z0-9_-]{18,}\\b)",
    re.IGNORECASE,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("weekly report input must be a JSON object")
    return payload


def _is_sensitive_string(value: str) -> bool:
    if ABSOLUTE_PATH_RE.match(value):
        return True
    return bool(SECRET_VALUE_RE.search(value))


def _ensure_safe_string(value: Any, *, field: str) -> Any:
    if isinstance(value, str) and _is_sensitive_string(value):
        raise ValueError(f"sanitized output would still contain a sensitive value in {field}")
    return value


def _sanitize_phase14(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    allowed = (
        "mode",
        "backfill_mode",
        "strict_gate_passed",
        "micro_state_frozen",
        "h_t",
        "rho_t",
        "k_hat_t",
        "s_t",
    )
    removed = 0
    sanitized: dict[str, Any] = {}
    for key in allowed:
        if key in payload:
            sanitized[key] = _ensure_safe_string(payload[key], field=f"phase14.{key}")
    removed += sum(1 for key in payload if key not in allowed)
    return sanitized, removed


def _sanitize_phase15(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    allowed = (
        "paper_only",
        "broker_submission_allowed",
        "signal_eligible",
        "execution_allowed",
        "orders_count",
        "reason",
        "delta_weights",
    )
    removed = 0
    sanitized: dict[str, Any] = {}
    for key in allowed:
        if key in payload:
            sanitized[key] = _ensure_safe_string(payload[key], field=f"phase15.{key}")
    removed += sum(1 for key in payload if key not in allowed)
    return sanitized, removed


def sanitize_report(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    required_top_level = ("week_end", "generated_at_utc", "system", "phase14", "phase15")
    for key in required_top_level:
        if key not in payload:
            raise ValueError(f"missing required field: {key}")

    sanitized_phase14, removed_phase14 = _sanitize_phase14(dict(payload["phase14"]))
    sanitized_phase15, removed_phase15 = _sanitize_phase15(dict(payload["phase15"]))

    removed_top_level = sum(1 for key in payload if key not in required_top_level)
    sanitized = {
        "week_end": _ensure_safe_string(payload["week_end"], field="week_end"),
        "generated_at_utc": _ensure_safe_string(payload["generated_at_utc"], field="generated_at_utc"),
        "system": _ensure_safe_string(payload["system"], field="system"),
        "phase14": sanitized_phase14,
        "phase15": sanitized_phase15,
        "sanitization": {
            "sanitized": True,
            "removed_fields_count": removed_top_level + removed_phase14 + removed_phase15,
            "policy": POLICY,
        },
    }

    for field, value in sanitized.items():
        if field == "sanitization":
            continue
        _scan_for_sensitive_values(value, field=field)
    return sanitized, removed_top_level + removed_phase14 + removed_phase15


def _scan_for_sensitive_values(value: Any, *, field: str) -> None:
    if isinstance(value, str):
        if _is_sensitive_string(value):
            raise ValueError(f"sanitized output would contain a sensitive value in {field}")
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            _scan_for_sensitive_values(nested, field=f"{field}.{key}")
        return
    if isinstance(value, list):
        for idx, nested in enumerate(value):
            _scan_for_sensitive_values(nested, field=f"{field}[{idx}]")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        payload = _load_json(args.input)
        sanitized, _removed = sanitize_report(payload)
        _write_json(args.output, sanitized)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"sanitized report written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
