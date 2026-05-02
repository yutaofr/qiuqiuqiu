#!/bin/bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="$SCRIPT_ROOT"

# Ensure common macOS binary paths are in PATH for launchd compatibility
if [[ "$OSTYPE" == "darwin"* ]]; then
  export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
WEEK_END=""
NOW_UTC_OVERRIDE=""
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
    --now-utc)
      NOW_UTC_OVERRIDE="${2:-}"
      shift 2
      ;;
    --work-root)
      WORK_ROOT="${2:-}"
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
  WEEK_END="$("$PYTHON_BIN" - "${NOW_UTC_OVERRIDE:-${WEEKLY_ORCH_NOW_UTC:-}}" <<'PY'
from datetime import datetime, timedelta, time, timezone
from zoneinfo import ZoneInfo
import sys

raw = sys.argv[1] if len(sys.argv) > 1 else ""
if raw:
    now_utc = datetime.fromisoformat(raw.replace("Z", "+00:00"))
else:
    now_utc = datetime.now(timezone.utc)
if now_utc.tzinfo is None:
    now_utc = now_utc.replace(tzinfo=timezone.utc)
local = now_utc.astimezone(ZoneInfo("America/New_York"))
weekday = local.weekday()
date = local.date()
if weekday == 4:
    if local.time() >= time(16, 15):
        resolved = date
    else:
        resolved = date - timedelta(days=7)
elif weekday > 4:
    resolved = date - timedelta(days=weekday - 4)
else:
    resolved = date - timedelta(days=weekday + 3)
print(resolved.isoformat())
PY
)"
fi

if $RESEND && [[ -z "$RESEND_REASON" ]]; then
  echo "--resend requires --resend-reason" >&2
  exit 2
fi

if [[ -z "$WORK_ROOT" ]]; then
  WORK_ROOT="$SCRIPT_ROOT"
fi
WORK_ROOT="$(cd "$WORK_ROOT" && pwd)"
cd "$WORK_ROOT"

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
  local results
  results="$("$PYTHON_BIN" - "$WORK_ROOT/.env" <<'PY'
from pathlib import Path
import os
import stat
import sys

allowed = {"DISCORD_WEBHOOK_URL", "ALERT_WEBHOOK_URL", "GEMINI_CMD"}
env_path = Path(sys.argv[1])
mode = stat.S_IMODE(os.stat(env_path).st_mode)
if mode != 0o600:
    print(f"refusing to read {env_path}: permissions must be 0600, got {oct(mode)}", file=sys.stderr)
    raise SystemExit(1)
values = {}
for raw_line in env_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    key = key.strip()
    if key not in allowed:
        continue
    value = value.strip().strip("'\"")
    values[key] = value

# Output values in a way that bash can easily parse
if "DISCORD_WEBHOOK_URL" in values:
    print(f"DISCORD_WEBHOOK_URL={values['DISCORD_WEBHOOK_URL']}")
elif "ALERT_WEBHOOK_URL" in values:
    print(f"DISCORD_WEBHOOK_URL={values['ALERT_WEBHOOK_URL']}")

if "GEMINI_CMD" in values:
    print(f"GEMINI_CMD={values['GEMINI_CMD']}")
PY
)"
  if [[ -n "$results" ]]; then
    while IFS= read -r line; do
      export "$line"
    done <<< "$results"
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
    "你是一位资深的量化策略架构师，负责解读 QQQ 周期状态系统的周报数据。",
    "你的任务是根据下方的 JSON 数据，撰写一份简洁但深刻的中文市场洞察（Markdown 格式）。",
    "",
    "解读核心准则：",
    "1. 核心理论参考：",
    "   - s_t (State Stress): 宏观动能强度。数值越高，动能一致性越强。当前 0.9+ 代表极强趋势。",
    "   - h_t (Micro Fragility): 微观脆弱性。衡量持仓拥挤度，高值预示局部结构性风险。",
    "   - rho_t (Fracture Risk): 系统断裂风险（相关性同步）。反映流动性压力，不是概率。",
    "   - k_hat_t (Cycle Regime): 1-2=扩张期, 3=过热期, 4-5=回撤/调整期。",
    "2. 调仓细节解读：",
    "   - 如果存在 'delta_weights'，它代表投资组合权重的百分比变化（如 0.4 代表增仓 40%，-0.4 代表减仓 40%）。",
    "   - 请简要分析调仓的方向（如：从高 Beta 资产 QQQ 转向避险资产 BIL）。",
    "3. 信号解读要求：准确引用数值，基于上述理论进行逻辑推断。结合 'execution_permitted' 评估确定性。",
    "4. 语气：专业、精准、不带感情色彩。严禁空洞的宏观臆测。",
    "5. 格式约束：严禁使用 LaTeX 格式（如 $s_t$）。请使用纯文本（如 s_t）或行内代码。",
    "6. [私人理财顾问总结]: 在报告结尾增加此区块。请切换为私人理财顾问身份，用深入浅出的语言（如生活化类比）对当前处境做直觉化总结。请务必结合 'delta_weights' 给出具体的行动解释（如：‘我已经撤回了四成资金’）。严禁在总结中使用变量名或专业术语。",
    "",
    json.dumps(sanitized, ensure_ascii=False, indent=2, sort_keys=True),
    "",
    "输出要求：直接返回 Markdown 格式的洞察，标题使用 '### 周期状态观察'，内容控制在 3-5 个要点，末尾给出系统的最终结论（如：维持当前仓位、预警集中度风险等）。",
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
