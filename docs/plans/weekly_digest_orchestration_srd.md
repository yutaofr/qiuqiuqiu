# Weekly Digest Orchestration SRD

This document defines the minimum runnable orchestration entry points for the weekly digest flow.

## Scope

- `src/main.py`: generate a deterministic, machine-readable weekly report JSON.
- `scripts/sanitize_weekly_report.py`: allowlist sanitization for any report handed to Gemini or downstream automation.
- `src/output/send_insight.py`: Discord webhook sender for weekly insight and fallback error modes.
- `scripts/run_weekly_orchestration.sh`: Unix-style orchestration entrypoint with locking and idempotency markers.
- `launchd/com.qiuqiuqiu.weekly.plist.template`: launchd template for local scheduling.

## Hard Constraints

- Preserve Phase 14 and Phase 15 business semantics.
- Never connect to a real broker.
- Never bypass `paper_only=true` or `broker_submission_allowed=false`.
- Do not call Gemini or Discord in dry-run mode.
- Do not read `.env` directly with `source`.
- Do not log secrets.

## Required Artifacts

- `weekly_report.json`
- `weekly_report_sanitized.json`
- `gemini_prompt.md`
- `run_status.json`
- `sent_discord.ok`
- `notified_error_<stage>.ok`

## Minimum JSON Contract

The weekly report must include:

- `week_end`
- `generated_at_utc`
- `system`
- `source`
- `phase14`
- `phase15`

The sanitized report must use an allowlist schema and remove sensitive or non-portable fields.

## Operational Behavior

- Use the latest Phase 14 snapshot by default, or the matching historical snapshot when available.
- Use the latest Phase 15 sandbox summary by default.
- If the Phase 15 summary is missing, fail with a clear instruction to run the sandbox first.
- Build a Gemini prompt from the sanitized JSON only.
- On Gemini failure, send a deterministic fallback digest and mark the error stage once.
- On Discord success, write `sent_discord.ok` atomically.
- If `sent_discord.ok` exists and `--resend` is not provided, exit without duplicating the send.

## Scheduling Note

The launchd template is intentionally a template only. Operators must replace the absolute script path and configure the local timezone so that Friday 16:15 ET maps to the local calendar time.
