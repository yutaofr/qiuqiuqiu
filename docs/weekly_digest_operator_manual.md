# Weekly Digest Operator Manual

## Purpose

This workflow runs once per week on the local macOS host after Friday U.S. market close. It generates the weekly report, sanitizes it, asks local Gemini for an interpretation, and posts the result to Discord.

The workflow is paper-only. It does not submit broker orders.

## Inputs

- Local repository checkout
- `DISCORD_WEBHOOK_URL` in `.env` or the process environment
- Optional `WEEKLY_ORCH_NOW_UTC` or `--now-utc` for deterministic dry-runs

## Outputs

- `weekly_report.json`
- `weekly_report_sanitized.json`
- `gemini_prompt.md`
- `gemini_insight.md`
- `run_status.json`
- `sent_discord.ok`
- `notified_error_<stage>.ok`

## Manual Run

Dry-run:

```bash
bash scripts/run_weekly_orchestration.sh --dry-run --work-root /ABSOLUTE/PATH/TO/qiuqiuqiu
```

Pinned fake clock:

```bash
bash scripts/run_weekly_orchestration.sh \
  --dry-run \
  --work-root /ABSOLUTE/PATH/TO/qiuqiuqiu \
  --now-utc 2026-04-24T21:00:00Z
```

The `--week-end` flag is optional. When omitted, the script computes the latest completed Friday in `America/New_York`.

## Runtime Rules

- The sanitizer is mandatory.
- Gemini receives only sanitized JSON.
- `sent_discord.ok` prevents duplicate successful sends.
- `notified_error_<stage>.ok` prevents duplicate error spam.
- `--resend` requires `--resend-reason`.
- `.env` must be `0600` before it is read.

## Troubleshooting

- If the report step fails, check `outputs/phase14/cycle_snapshot_latest.json`.
- If the Phase 15 summary is missing, run `python scripts/run_phase15_sandbox.py --week-end YYYY-MM-DD`.
- If Discord delivery fails, inspect the weekly log under `logs/weekly/`.
- If the script rejects `.env`, fix the file mode to `0600`.

