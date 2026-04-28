"""Execute strict fixture and/or degraded real pipeline paths and write output artifacts.

Usage:
    python scripts/run_pipeline.py [--mode strict_fixture|degraded_real|both]

Output artifacts (written to outputs/pipeline/):
    strict_fixture_pipeline_output.csv
    degraded_real_pipeline_output.csv
    pipeline_mode_summary.json
    integration_acceptance.md
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

# Allow running from repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qqq_cycle.data_contracts.constituents import CsvConstituentStore
from qqq_cycle.data_contracts.pit_adjustment import CsvPITAdjustmentEngine
from qqq_cycle.data_contracts.weights import CsvWeightStore
from qqq_cycle.pipeline import (
    MODE_DEGRADED,
    MODE_STRICT,
    MODE_WARMUP,
    PipelineContracts,
    PipelineResult,
    results_to_frame,
    run_pipeline,
)
from tests.fixtures.strict_pipeline_fixture import (
    make_strict_contracts,
    make_strict_macro_inputs,
    run_strict_fixture,
)

OUTPUT_DIR = Path("outputs/pipeline")
REAL_STAGING_CSV = Path("cache/real_replay/staging/weekly_inputs.csv")
MICRO_CACHE_DIR = Path("cache/micro")


def _write_csv(results: list[PipelineResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    results_to_frame(results).to_csv(path, index=False)
    print(f"  wrote {path} ({len(results)} rows)")


def _first_date_with_mode(results: list[PipelineResult], modes: set[str]) -> str | None:
    for r in results:
        if r.mode in modes:
            return r.week_end
    return None


def _run_strict_fixture_path() -> tuple[bool, list[str], list[PipelineResult]]:
    """Run strict fixture path. Returns (passed, blockers, results)."""
    blockers: list[str] = []
    try:
        results = run_strict_fixture()
    except Exception as exc:
        return False, [f"run_strict_fixture raised: {exc}"], []

    strict_rows = [r for r in results if r.mode == MODE_STRICT]
    if not strict_rows:
        blockers.append("no strict rows produced")
    else:
        null_h = [r for r in strict_rows if r.h_t is None]
        null_rho = [r for r in strict_rows if r.rho_t is None]
        if null_h:
            blockers.append(f"{len(null_h)} strict rows have h_t=null")
        if null_rho:
            blockers.append(f"{len(null_rho)} strict rows have rho_t=null")
        bad_reason = [r for r in strict_rows if r.degraded_reason is not None]
        if bad_reason:
            blockers.append(f"{len(bad_reason)} strict rows have non-null degraded_reason")

    return len(blockers) == 0, blockers, results


def _run_strict_real_path() -> tuple[bool, list[str], list[PipelineResult]]:
    """Run strict real path using CSV micro stores. Returns (passed, blockers, results)."""
    blockers: list[str] = []

    if not REAL_STAGING_CSV.exists():
        return False, [f"staging CSV not found: {REAL_STAGING_CSV}"], []
    if not (MICRO_CACHE_DIR / "constituents.csv").exists():
        return False, ["cache/micro/constituents.csv not found — run scripts/seed_micro_data.py"], []

    try:
        inputs = pd.read_csv(REAL_STAGING_CSV, index_col=0, parse_dates=True)
        inputs.index = pd.to_datetime(inputs.index)
    except Exception as exc:
        return False, [f"failed to load staging CSV: {exc}"], []

    try:
        pit_engine = CsvPITAdjustmentEngine(MICRO_CACHE_DIR / "prices")
        constituent_store = CsvConstituentStore(MICRO_CACHE_DIR / "constituents.csv")
        weight_store = CsvWeightStore(MICRO_CACHE_DIR / "weights.csv")
    except Exception as exc:
        return False, [f"failed to load micro stores: {exc}"], []

    contracts = PipelineContracts(
        pit_engine=pit_engine,
        constituent_store=constituent_store,
        weight_store=weight_store,
    )

    try:
        results = run_pipeline(inputs, contracts=contracts)
    except Exception as exc:
        traceback.print_exc()
        return False, [f"run_pipeline raised: {exc}"], []

    post_warmup = [r for r in results if r.mode != MODE_WARMUP]
    if not post_warmup:
        blockers.append("no post-warmup rows produced")
        return False, blockers, results

    strict_rows = [r for r in post_warmup if r.mode == MODE_STRICT]
    if not strict_rows:
        blockers.append(
            "no strict rows produced — micro data coverage may not overlap with "
            "post-warmup period; check that seed data range has ≥156 weeks of data"
        )
    else:
        null_h = [r for r in strict_rows if r.h_t is None]
        null_rho = [r for r in strict_rows if r.rho_t is None]
        if null_h:
            blockers.append(f"{len(null_h)} strict rows have h_t=null")
        if null_rho:
            blockers.append(f"{len(null_rho)} strict rows have rho_t=null")

    return len(blockers) == 0, blockers, results


def _run_degraded_real_path() -> tuple[bool, list[str], list[str], list[PipelineResult]]:
    """Run degraded real path. Returns (passed, blockers, degraded_reasons, results)."""
    blockers: list[str] = []
    degraded_reasons: list[str] = []

    if not REAL_STAGING_CSV.exists():
        return False, [f"staging CSV not found: {REAL_STAGING_CSV}"], [], []

    try:
        inputs = pd.read_csv(REAL_STAGING_CSV, index_col=0, parse_dates=True)
        inputs.index = pd.to_datetime(inputs.index)
    except Exception as exc:
        return False, [f"failed to load staging CSV: {exc}"], [], []

    try:
        results = run_pipeline(inputs, contracts=None)
    except Exception as exc:
        return False, [f"run_pipeline raised: {exc}"], [], []

    post_warmup = [r for r in results if r.mode != MODE_WARMUP]
    if not post_warmup:
        blockers.append("no post-warmup rows produced")
        return False, blockers, degraded_reasons, results

    # h_t and rho_t must be null for all post-warmup rows (no contracts provided).
    not_null_h = [r for r in post_warmup if r.h_t is not None]
    not_null_rho = [r for r in post_warmup if r.rho_t is not None]
    if not_null_h:
        blockers.append(f"{len(not_null_h)} post-warmup rows have non-null h_t (strict leak)")
    if not_null_rho:
        blockers.append(f"{len(not_null_rho)} post-warmup rows have non-null rho_t (strict leak)")

    # degraded_reason must be non-null for every degraded row.
    missing_reason = [r for r in post_warmup if r.mode == MODE_DEGRADED and not r.degraded_reason]
    if missing_reason:
        blockers.append(f"{len(missing_reason)} degraded rows have null/empty degraded_reason (silent fallback)")

    # state/stress outputs must be present on at least some post-warmup rows.
    has_state = any(r.k_hat_t is not None for r in post_warmup)
    has_stress = any(r.s_t is not None for r in post_warmup)
    if not has_state:
        blockers.append("no post-warmup rows have k_hat_t (state engine not running)")
    if not has_stress:
        blockers.append("no post-warmup rows have s_t (stress engine not running)")

    degraded_reasons = sorted(
        set(r.degraded_reason for r in post_warmup if r.degraded_reason)
    )
    return len(blockers) == 0, blockers, degraded_reasons, results


def _write_summary(
    strict_passed: bool,
    degraded_passed: bool,
    strict_real_passed: bool,
    strict_blockers: list[str],
    degraded_reasons: list[str],
    strict_real_blockers: list[str],
    strict_results: list[PipelineResult],
    degraded_results: list[PipelineResult],
    strict_real_results: list[PipelineResult],
    output_dir: Path,
) -> None:
    reference_results = degraded_results or strict_results or strict_real_results
    first_state = _first_date_with_mode(reference_results, {MODE_STRICT, MODE_DEGRADED})
    first_stress = first_state

    summary = {
        "strict_fixture_passed": strict_passed,
        "degraded_real_passed": degraded_passed,
        "real_strict_pipeline_passed": strict_real_passed,
        "strict_blockers": strict_blockers,
        "degraded_reasons": degraded_reasons,
        "strict_real_blockers": strict_real_blockers,
        "first_valid_state_date": first_state,
        "first_valid_stress_date": first_stress,
    }
    path = output_dir / "pipeline_mode_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {path}")


def _write_acceptance(
    strict_passed: bool,
    degraded_passed: bool,
    strict_real_passed: bool,
    strict_blockers: list[str],
    degraded_reasons: list[str],
    strict_real_blockers: list[str],
    strict_results: list[PipelineResult],
    degraded_results: list[PipelineResult],
    strict_real_results: list[PipelineResult],
    output_dir: Path,
) -> None:
    lines = ["# Phase 6 Integration Acceptance Report\n"]

    def _status(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    lines.append("## Outcome A — Strict Fixture Path\n")
    lines.append(f"**Status: {_status(strict_passed)}**\n")
    if strict_results:
        strict_rows = [r for r in strict_results if r.mode == MODE_STRICT]
        warmup_rows = [r for r in strict_results if r.mode == MODE_WARMUP]
        lines.append(f"- Total rows: {len(strict_results)}")
        lines.append(f"- Warmup rows: {len(warmup_rows)}")
        lines.append(f"- Strict rows: {len(strict_rows)}")
        if strict_rows:
            null_h = sum(1 for r in strict_rows if r.h_t is None)
            null_rho = sum(1 for r in strict_rows if r.rho_t is None)
            lines.append(f"- h_t null in strict rows: {null_h} (expected 0)")
            lines.append(f"- rho_t null in strict rows: {null_rho} (expected 0)")
    if strict_blockers:
        lines.append("\n**Blockers:**")
        for b in strict_blockers:
            lines.append(f"- {b}")
    lines.append("")

    lines.append("## Outcome B — Real Degraded Path\n")
    lines.append(f"**Status: {_status(degraded_passed)}**\n")
    if degraded_results:
        post_warmup = [r for r in degraded_results if r.mode != MODE_WARMUP]
        warmup_rows = [r for r in degraded_results if r.mode == MODE_WARMUP]
        degraded_rows = [r for r in degraded_results if r.mode == MODE_DEGRADED]
        lines.append(f"- Total rows: {len(degraded_results)}")
        lines.append(f"- Warmup rows: {len(warmup_rows)}")
        lines.append(f"- Degraded rows: {len(degraded_rows)}")
        if post_warmup:
            has_h = sum(1 for r in post_warmup if r.h_t is not None)
            has_rho = sum(1 for r in post_warmup if r.rho_t is not None)
            has_state = sum(1 for r in post_warmup if r.k_hat_t is not None)
            lines.append(f"- h_t non-null post-warmup: {has_h} (expected 0)")
            lines.append(f"- rho_t non-null post-warmup: {has_rho} (expected 0)")
            lines.append(f"- k_hat_t non-null post-warmup: {has_state} (expected >0)")
    if degraded_reasons:
        lines.append("\n**Degraded reasons observed:**")
        for r in degraded_reasons:
            lines.append(f"- {r}")
    lines.append("")

    lines.append("## Outcome C — Real Strict Path\n")
    lines.append(f"**Status: {_status(strict_real_passed)}**\n")
    if strict_real_results:
        sr_strict = [r for r in strict_real_results if r.mode == MODE_STRICT]
        sr_warmup = [r for r in strict_real_results if r.mode == MODE_WARMUP]
        sr_degraded = [r for r in strict_real_results if r.mode == MODE_DEGRADED]
        lines.append(f"- Total rows: {len(strict_real_results)}")
        lines.append(f"- Warmup rows: {len(sr_warmup)}")
        lines.append(f"- Strict rows: {len(sr_strict)}")
        lines.append(f"- Degraded rows: {len(sr_degraded)}")
        if sr_strict:
            null_h = sum(1 for r in sr_strict if r.h_t is None)
            null_rho = sum(1 for r in sr_strict if r.rho_t is None)
            lines.append(f"- h_t null in strict rows: {null_h} (expected 0)")
            lines.append(f"- rho_t null in strict rows: {null_rho} (expected 0)")
    if strict_real_blockers:
        lines.append("\n**Blockers:**")
        for b in strict_real_blockers:
            lines.append(f"- {b}")
    lines.append("")

    lines.append("## Acceptance Criteria Checklist\n")
    criteria = [
        ("strict_fixture_pipeline_output.csv exists", bool(strict_results)),
        ("degraded_real_pipeline_output.csv exists", bool(degraded_results)),
        ("strict_real_pipeline_output.csv exists", bool(strict_real_results)),
        ("fixture strict rows: h_t non-null", strict_passed and any(
            r.h_t is not None for r in strict_results if r.mode == MODE_STRICT
        )),
        ("fixture strict rows: rho_t non-null", strict_passed and any(
            r.rho_t is not None for r in strict_results if r.mode == MODE_STRICT
        )),
        ("degraded post-warmup: h_t all null", degraded_passed and all(
            r.h_t is None for r in degraded_results if r.mode != MODE_WARMUP
        )),
        ("degraded rows: degraded_reason non-null", degraded_passed and all(
            r.degraded_reason for r in degraded_results if r.mode == MODE_DEGRADED
        )),
        ("real strict rows: h_t non-null", strict_real_passed and any(
            r.h_t is not None for r in strict_real_results if r.mode == MODE_STRICT
        )),
        ("real strict rows: rho_t non-null", strict_real_passed and any(
            r.rho_t is not None for r in strict_real_results if r.mode == MODE_STRICT
        )),
        ("pipeline_mode_summary.json written", True),
    ]
    for criterion, ok in criteria:
        lines.append(f"- [{_status(ok)}] {criterion}")

    path = output_dir / "integration_acceptance.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 6/7 pipeline paths")
    parser.add_argument(
        "--mode",
        choices=["strict_fixture", "degraded_real", "strict_real", "both", "all"],
        default="both",
        help="'both' runs strict_fixture+degraded_real; 'all' adds strict_real",
    )
    args = parser.parse_args()

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    strict_passed = False
    degraded_passed = False
    strict_real_passed = False
    strict_blockers: list[str] = []
    degraded_reasons: list[str] = []
    strict_real_blockers: list[str] = []
    strict_results: list[PipelineResult] = []
    degraded_results: list[PipelineResult] = []
    strict_real_results: list[PipelineResult] = []

    if args.mode in ("strict_fixture", "both", "all"):
        print("\n=== Strict Fixture Path ===")
        strict_passed, strict_blockers, strict_results = _run_strict_fixture_path()
        if strict_results:
            _write_csv(strict_results, output_dir / "strict_fixture_pipeline_output.csv")
        print(f"  result: {'PASS' if strict_passed else 'FAIL'}")
        if strict_blockers:
            for b in strict_blockers:
                print(f"  BLOCKER: {b}")

    if args.mode in ("degraded_real", "both", "all"):
        print("\n=== Degraded Real Path ===")
        degraded_passed, _blockers, degraded_reasons, degraded_results = _run_degraded_real_path()
        if degraded_results:
            _write_csv(degraded_results, output_dir / "degraded_real_pipeline_output.csv")
        print(f"  result: {'PASS' if degraded_passed else 'FAIL'}")
        if _blockers:
            for b in _blockers:
                print(f"  BLOCKER: {b}")

    if args.mode in ("strict_real", "all"):
        print("\n=== Strict Real Path ===")
        strict_real_passed, strict_real_blockers, strict_real_results = _run_strict_real_path()
        if strict_real_results:
            _write_csv(strict_real_results, output_dir / "strict_real_pipeline_output.csv")
        print(f"  result: {'PASS' if strict_real_passed else 'FAIL'}")
        if strict_real_blockers:
            for b in strict_real_blockers:
                print(f"  BLOCKER: {b}")

    print("\n=== Writing Summary Artifacts ===")
    _write_summary(
        strict_passed, degraded_passed, strict_real_passed,
        strict_blockers, degraded_reasons, strict_real_blockers,
        strict_results, degraded_results, strict_real_results,
        output_dir,
    )
    _write_acceptance(
        strict_passed, degraded_passed, strict_real_passed,
        strict_blockers, degraded_reasons, strict_real_blockers,
        strict_results, degraded_results, strict_real_results,
        output_dir,
    )

    print(f"\nStrict fixture: {'PASS' if strict_passed else 'FAIL'}")
    print(f"Degraded real:  {'PASS' if degraded_passed else 'FAIL'}")
    print(f"Strict real:    {'PASS' if strict_real_passed else 'FAIL (not run)' if not strict_real_results else 'FAIL'}")

    if args.mode == "both" and (not strict_passed or not degraded_passed):
        sys.exit(1)
    if args.mode == "all" and (not strict_passed or not degraded_passed or not strict_real_passed):
        sys.exit(1)
    if args.mode == "strict_fixture" and not strict_passed:
        sys.exit(1)
    if args.mode == "degraded_real" and not degraded_passed:
        sys.exit(1)
    if args.mode == "strict_real" and not strict_real_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
