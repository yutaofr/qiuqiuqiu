#!/bin/bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WEEK_END=""
DRY_RUN=false
RERUN=false
RESEND=false
RESEND_REASON=""
RECOVER_STALE_LOCK=false
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --week-end)
      WEEK_END="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --rerun)
      RERUN=true
      shift
      ;;
    --resend)
      RESEND=true
      shift
      ;;
    --resend-reason)
      RESEND_REASON="${2:-}"
      shift 2
      ;;
    --recover-stale-lock)
      RECOVER_STALE_LOCK=true
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$WEEK_END" ]]; then
  echo "--week-end is required" >&2
  exit 2
fi

if $RESEND && [[ -z "$RESEND_REASON" ]]; then
  echo "--resend requires --resend-reason" >&2
  exit 2
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="$WORK_ROOT/.temp/weekly/$RUN_ID"
if [[ -n "$OUTPUT_DIR" ]]; then
  WEEK_ROOT="$OUTPUT_DIR"
elif $DRY_RUN; then
  WEEK_ROOT="$WORK_ROOT/.temp/weekly"
else
  WEEK_ROOT="$WORK_ROOT/outputs/weekly"
fi
WEEK_DIR="$WEEK_ROOT/$WEEK_END"
LOG="$WORK_ROOT/logs/weekly/$RUN_ID.log"
LOCK_DIR="$WEEK_DIR/.run_lock"
STATUS_PATH="$WEEK_DIR/run_status.json"
REPORT_JSON="$RUN_DIR/weekly_report.json"
SANITIZED_JSON="$RUN_DIR/weekly_report_sanitized.json"
PROMPT_PATH="$RUN_DIR/gemini_prompt.md"
GEMINI_OUTPUT="$RUN_DIR/gemini_insight.md"

mkdir -p "$(dirname "$LOG")" "$RUN_DIR" "$WEEK_DIR"
: >"$LOG"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$LOG"
}

write_marker_atomic() {
  local target="$1"
  local tmp="${target}.tmp.$$"
  printf 'ok\n' >"$tmp"
  mv "$tmp" "$target"
}

write_status() {
  local success="$1"
  local stage="$2"
  local message="$3"
  local tmp="${STATUS_PATH}.tmp.$$"
  "$PYTHON_BIN" - "$tmp" "$success" "$stage" "$message" "$WEEK_END" "$RUN_ID" "$RUN_DIR" "$REPORT_JSON" "$SANITIZED_JSON" "$PROMPT_PATH" "$GEMINI_OUTPUT" "$DRY_RUN" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
payload = {
    "week_end": sys.argv[5],
    "run_id": sys.argv[6],
    "run_dir": sys.argv[7],
    "report_json": sys.argv[8],
    "sanitized_json": sys.argv[9],
    "prompt_path": sys.argv[10],
    "gemini_output": sys.argv[11],
    "dry_run": sys.argv[12] == "true",
    "success": sys.argv[2] == "true",
    "stage": sys.argv[3],
    "message": sys.argv[4],
}
out.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  mv "$tmp" "$STATUS_PATH"
}

load_safe_env() {
  if [[ ! -f "$WORK_ROOT/.env" ]]; then
    return 0
  fi
  local webhook
  webhook="$("$PYTHON_BIN" - "$WORK_ROOT/.env" <<'PY'
from pathlib import Path
import sys

allowed = {"DISCORD_WEBHOOK_URL"}
env_path = Path(sys.argv[1])
for raw_line in env_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    key = key.strip()
    if key not in allowed:
        continue
    value = value.strip().strip("'\"")
    print(value)
    break
PY
)"
  if [[ -n "$webhook" ]]; then
    export DISCORD_WEBHOOK_URL="$webhook"
  fi
}

run_with_timeout() {
  local command_string="$1"
  local input_path="$2"
  local output_path="$3"
  local stderr_log="$4"
  "$PYTHON_BIN" - "$command_string" "$input_path" "$output_path" "$stderr_log" <<'PY'
import pathlib
import subprocess
import sys

cmd = sys.argv[1]
input_path = pathlib.Path(sys.argv[2])
output_path = pathlib.Path(sys.argv[3])
stderr_log = pathlib.Path(sys.argv[4])
prompt = input_path.read_text(encoding="utf-8")
result = subprocess.run(
    cmd,
    shell=True,
    input=prompt,
    text=True,
    capture_output=True,
    timeout=120,
)
output_path.write_text(result.stdout, encoding="utf-8")
if result.stderr:
    with stderr_log.open("a", encoding="utf-8") as handle:
        handle.write(result.stderr)
raise SystemExit(result.returncode)
PY
}

fail_with_status() {
  local stage="$1"
  local message="$2"
  log "stage=$stage status=failed message=$message"
  write_status false "$stage" "$message"
  exit 1
}

cleanup() {
  rm -rf "$LOCK_DIR"
}

if mkdir "$LOCK_DIR" 2>/dev/null; then
  trap cleanup EXIT
else
  if ! $RECOVER_STALE_LOCK; then
    echo "run_lock already exists for $WEEK_END; pass --recover-stale-lock to reclaim it" >&2
    exit 1
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
  trap cleanup EXIT
fi

load_safe_env

if [[ -f "$WEEK_DIR/sent_discord.ok" && $RESEND == false && $RERUN == false ]]; then
  log "stage=sent_discord status=skipped reason=existing_marker path=$WEEK_DIR/sent_discord.ok"
  exit 0
fi

if ! "$PYTHON_BIN" "$SCRIPT_ROOT/src/main.py" --week-end "$WEEK_END" --output "$REPORT_JSON" >>"$LOG" 2>&1; then
  fail_with_status "main_report" "weekly report generation failed"
fi
log "stage=main_report path=$REPORT_JSON"

if ! "$PYTHON_BIN" -m json.tool "$REPORT_JSON" >/dev/null; then
  fail_with_status "json_validation" "weekly report JSON validation failed"
fi
log "stage=json_validation path=$REPORT_JSON"

if ! "$PYTHON_BIN" "$SCRIPT_ROOT/scripts/sanitize_weekly_report.py" --input "$REPORT_JSON" --output "$SANITIZED_JSON" >>"$LOG" 2>&1; then
  fail_with_status "sanitizer" "weekly report sanitization failed"
fi
log "stage=sanitizer path=$SANITIZED_JSON"

"$PYTHON_BIN" - "$SANITIZED_JSON" "$PROMPT_PATH" <<'PY' >>"$LOG" 2>&1
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
dest = pathlib.Path(sys.argv[2])
sanitized = json.loads(source.read_text(encoding="utf-8"))
prompt = [
    "You are drafting the weekly digest insight.",
    "Use only the sanitized JSON below.",
    "",
    json.dumps(sanitized, ensure_ascii=True, indent=2, sort_keys=True),
    "",
    "Return a concise markdown insight.",
]
dest.write_text("\n".join(prompt) + "\n", encoding="utf-8")
PY
log "stage=prompt_build path=$PROMPT_PATH"

if $DRY_RUN; then
  log "stage=dry_run status=success"
  write_status true "dry_run" "dry-run completed without Gemini or Discord"
  exit 0
fi

if [[ -z "${GEMINI_CMD:-}" ]]; then
  fail_with_status "gemini" "GEMINI_CMD is required for non-dry-run orchestration"
fi

if ! run_with_timeout "$GEMINI_CMD" "$PROMPT_PATH" "$GEMINI_OUTPUT" "$LOG"; then
  if [[ ! -f "$WEEK_DIR/notified_error_gemini.ok" ]]; then
    if ! "$PYTHON_BIN" "$SCRIPT_ROOT/src/output/send_insight.py" --mode fallback_error --stage gemini --validated-json "$SANITIZED_JSON" --message "[ERROR] AI Interpretation Failed." >>"$LOG" 2>&1; then
      fail_with_status "gemini" "fallback notification failed after Gemini failure"
    fi
    write_marker_atomic "$WEEK_DIR/notified_error_gemini.ok"
    log "stage=fallback_error path=$WEEK_DIR/notified_error_gemini.ok"
  else
    log "stage=fallback_error status=skipped reason=existing_marker path=$WEEK_DIR/notified_error_gemini.ok"
  fi
  write_status false "gemini" "Gemini invocation failed"
  exit 1
fi

if [[ ! -s "$GEMINI_OUTPUT" ]]; then
  if [[ ! -f "$WEEK_DIR/notified_error_gemini.ok" ]]; then
    if ! "$PYTHON_BIN" "$SCRIPT_ROOT/src/output/send_insight.py" --mode fallback_error --stage gemini --validated-json "$SANITIZED_JSON" --message "[ERROR] AI Interpretation Failed." >>"$LOG" 2>&1; then
      fail_with_status "gemini" "fallback notification failed after empty Gemini output"
    fi
    write_marker_atomic "$WEEK_DIR/notified_error_gemini.ok"
    log "stage=fallback_error path=$WEEK_DIR/notified_error_gemini.ok"
  else
    log "stage=fallback_error status=skipped reason=existing_marker path=$WEEK_DIR/notified_error_gemini.ok"
  fi
  write_status false "gemini" "Gemini produced empty output"
  exit 1
fi
log "stage=gemini_output path=$GEMINI_OUTPUT"

if [[ -f "$WEEK_DIR/sent_discord.ok" && $RESEND == false ]]; then
  log "stage=discord status=skipped reason=existing_marker path=$WEEK_DIR/sent_discord.ok"
  write_status true "sent_marker" "local artifacts rebuilt; Discord send skipped due to existing marker"
  exit 0
fi

if ! "$PYTHON_BIN" "$SCRIPT_ROOT/src/output/send_insight.py" --mode insight --input "$GEMINI_OUTPUT" >>"$LOG" 2>&1; then
  fail_with_status "discord" "Discord insight send failed"
fi
write_marker_atomic "$WEEK_DIR/sent_discord.ok"
log "stage=discord path=$WEEK_DIR/sent_discord.ok"
write_status true "success" "weekly orchestration completed successfully"
exit 0
