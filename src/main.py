"""Generate the weekly digest report JSON for orchestration.

The entrypoint is intentionally side-effect free except for writing the report
to the path supplied via ``--output``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE14_LATEST = Path("outputs/phase14/cycle_snapshot_latest.json")
PHASE15_LATEST = Path("outputs/phase15/execution_sandbox_summary_latest.json")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _generated_at_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_phase14_snapshot(week_end: str) -> Path:
    latest = PHASE14_LATEST
    if latest.exists():
        try:
            payload = _load_json(latest)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard.
            raise ValueError(f"invalid Phase 14 snapshot JSON at {latest}") from exc
        if str(payload.get("week_end")) == week_end:
            return latest

    history_dir = Path("outputs/phase14/history")
    if history_dir.exists():
        matches = sorted(history_dir.glob(f"cycle_snapshot_{week_end}__run_*.json"))
        if matches:
            return matches[-1]

    if latest.exists():
        return latest

    raise FileNotFoundError(f"no Phase 14 snapshot found for week_end={week_end}")


def _resolve_phase15_summary(week_end: str) -> tuple[Path, Path]:
    summary_path = PHASE15_LATEST
    delta_path = Path(f"outputs/phase15/portfolio_delta_{week_end}.json")
    if not summary_path.exists():
        raise FileNotFoundError(
            "Phase 15 summary missing; run python scripts/run_phase15_sandbox.py --week-end "
            f"{week_end} first"
        )
    if not delta_path.exists():
         # Fallback to latest if week-end specific missing (e.g. initial run)
         delta_path = Path("outputs/phase15/portfolio_delta_latest.json")

    return summary_path, delta_path


def _phase14_section(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "snapshot_path": str(path),
        "mode": payload.get("mode"),
        "backfill_mode": payload.get("backfill_mode"),
        "strict_gate_passed": bool(payload.get("strict_gate_passed", False)),
        "micro_state_frozen": bool(payload.get("micro_state_frozen", False)),
        "h_t": payload.get("h_t"),
        "rho_t": payload.get("rho_t"),
        "k_hat_t": payload.get("k_hat_t"),
        "s_t": payload.get("s_t"),
        "source_hash": payload.get("source_hash"),
    }


def _phase15_section(path: Path, payload: dict[str, Any], delta_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    section = {
        "summary_path": str(path),
        "paper_only": bool(payload.get("paper_only", False)),
        "broker_submission_allowed": bool(payload.get("broker_submission_allowed", False)),
        "signal_eligible": bool(payload.get("signal_eligible", False)),
        "execution_allowed": bool(payload.get("execution_allowed", False)),
        "orders_count": int(payload.get("orders_count", 0)),
        "reason": payload.get("reason"),
    }
    if delta_payload:
        section["delta_weights"] = delta_payload.get("delta_weights", {})
    return section


def build_weekly_report(week_end: str) -> dict[str, Any]:
    phase14_path = _resolve_phase14_snapshot(week_end)
    summary_path, delta_path = _resolve_phase15_summary(week_end)
    phase14_payload = _load_json(phase14_path)
    phase15_payload = _load_json(summary_path)

    delta_payload = None
    if delta_path.exists():
        delta_payload = _load_json(delta_path)

    return {
        "week_end": week_end,
        "generated_at_utc": _generated_at_utc(),
        "system": "qiuqiuqiu",
        "source": "weekly_digest",
        "phase14": _phase14_section(phase14_path, phase14_payload),
        "phase15": _phase15_section(summary_path, phase15_payload, delta_payload),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week-end", required=True, help="Decision week end in YYYY-MM-DD format.")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the weekly report JSON.")
    args = parser.parse_args(argv)

    try:
        datetime.strptime(args.week_end, "%Y-%m-%d")
    except ValueError:
        print(f"invalid --week-end value: {args.week_end}", file=sys.stderr)
        return 2

    try:
        report = build_weekly_report(args.week_end)
        _write_json(args.output, report)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"failed to parse report source JSON: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive fallback.
        print(f"failed to build weekly report: {exc}", file=sys.stderr)
        return 1

    print(f"weekly report written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
