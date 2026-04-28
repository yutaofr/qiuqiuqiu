"""T12: Strict real coverage boundary tests.

These tests verify:
1. No strict row exceeds the micro source coverage end date.
2. Constituent ticker universe is bounded by the seed manifest.
3. Pipeline cuts to degraded (not forward-fills strict) after micro coverage ends.
4. pipeline_mode_summary.json contains all required coverage metadata fields with
   correct values (including production_eligible=false, grade=conditional).

All tests skip if cache/micro/constituents.csv or the required output files
do not exist.  Run scripts/seed_micro_data.py and
`python scripts/run_pipeline.py --mode all` to generate them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from qqq_cycle.pipeline import MODE_DEGRADED, MODE_STRICT, MODE_WARMUP

_MICRO_DIR = Path("cache/micro")
_STRICT_REAL_CSV = Path("outputs/pipeline/strict_real_pipeline_output.csv")
_MANIFEST = _MICRO_DIR / "seed_manifest.json"
_SUMMARY_JSON = Path("outputs/pipeline/pipeline_mode_summary.json")
_SKIP_MICRO = "micro seed data not found; run scripts/seed_micro_data.py"
_SKIP_OUTPUTS = "pipeline outputs not found; run scripts/run_pipeline.py --mode all"


def _skip_if_no_micro():
    return pytest.mark.skipif(
        not (_MICRO_DIR / "constituents.csv").exists(), reason=_SKIP_MICRO
    )


def _skip_if_no_outputs():
    return pytest.mark.skipif(
        not _STRICT_REAL_CSV.exists(), reason=_SKIP_OUTPUTS
    )


# ── T12.1: no strict row exceeds micro source end ────────────────────────────

@_skip_if_no_micro()
@_skip_if_no_outputs()
def test_strict_rows_do_not_exceed_micro_source_end():
    """Strict row week_end must not exceed micro source end + 7 days.

    The seed manifest's 'end' date is the last date for which micro data is
    available.  Weekly Friday labels may land up to 6 days after the last
    trading day, so we allow a 7-day slack.  Any strict row beyond that
    would be produced from data that does not exist.
    """
    if not _MANIFEST.exists():
        pytest.skip("seed_manifest.json not found")

    manifest = json.loads(_MANIFEST.read_text())
    micro_end = pd.Timestamp(manifest["end"])
    max_allowed = micro_end + pd.Timedelta(days=7)

    df = pd.read_csv(_STRICT_REAL_CSV)
    strict_df = df[df["mode"] == MODE_STRICT].copy()

    if strict_df.empty:
        pytest.skip("no strict rows in output")

    strict_weeks = pd.to_datetime(strict_df["week_end"])
    violations = strict_weeks[strict_weeks > max_allowed]
    assert violations.empty, (
        f"Strict rows exceed micro source end ({micro_end.date()} + 7 days = "
        f"{max_allowed.date()}):\n"
        + "\n".join(f"  {d.date()}" for d in sorted(violations))
    )


# ── T12.2: constituent ticker universe bounded by seed manifest ───────────────

@_skip_if_no_micro()
def test_constituent_tickers_bounded_by_seed_manifest():
    """All tickers in the constituent store must be in the seed manifest.

    Tickers outside the seed manifest have no price data in CsvPITAdjustmentEngine.
    They would silently produce zero-weight contributions and pollute micro
    layer results without raising an error.
    """
    if not _MANIFEST.exists():
        pytest.skip("seed_manifest.json not found")

    from qqq_cycle.data_contracts.constituents import CsvConstituentStore

    manifest = json.loads(_MANIFEST.read_text())
    seed_tickers = set(manifest.get("tickers_successful", []))

    cs = CsvConstituentStore(_MICRO_DIR / "constituents.csv")
    all_tickers = set(cs._df["ticker"].unique())

    unknown = all_tickers - seed_tickers
    assert not unknown, (
        f"Constituent store contains tickers absent from seed manifest: {sorted(unknown)}\n"
        "These tickers have no price data and would produce zero-weight contributions."
    )


# ── T12.3: pipeline cuts to degraded after micro coverage ends ────────────────

@_skip_if_no_micro()
@_skip_if_no_outputs()
def test_pipeline_cuts_to_degraded_after_strict_coverage_ends():
    """Rows after the last strict week must be degraded, not forward-filled strict.

    If micro data is exhausted mid-series, the pipeline must switch to degraded
    mode.  Strict rows cannot be forward-filled beyond the last week for which
    micro data satisfies the z_wrob_156 window.
    """
    df = pd.read_csv(_STRICT_REAL_CSV)
    df["week_end"] = pd.to_datetime(df["week_end"])

    strict_df = df[df["mode"] == MODE_STRICT]
    post_warmup_df = df[df["mode"] != MODE_WARMUP]

    if strict_df.empty:
        pytest.skip("no strict rows in output; cannot test handoff")

    last_strict_week = strict_df["week_end"].max()
    later_rows = post_warmup_df[post_warmup_df["week_end"] > last_strict_week]

    if later_rows.empty:
        # Pipeline ends at last strict week — no trailing rows to inspect.
        return

    non_degraded = later_rows[later_rows["mode"] != MODE_DEGRADED]
    assert non_degraded.empty, (
        f"Found {len(non_degraded)} non-degraded rows after last strict week "
        f"({last_strict_week.date()}) — strict output must not be forward-filled:\n"
        + non_degraded[["week_end", "mode"]].to_string()
    )

    missing_reason = later_rows[
        later_rows["degraded_reason"].isna() | (later_rows["degraded_reason"].astype(str) == "")
    ]
    assert missing_reason.empty, (
        f"Found {len(missing_reason)} post-strict-coverage degraded rows with null/empty reason"
    )


# ── T12.4: summary metadata fields present and internally consistent ──────────

@_skip_if_no_micro()
def test_summary_metadata_fields_present_and_consistent():
    """pipeline_mode_summary.json must contain all required coverage metadata.

    Field values must satisfy hard constraints:
    - phase_7_verdict == 'pass_conditional'
    - production_strict_pipeline_passed == False
    - strict_real_production_eligible == False
    - strict_real_contract_grade == 'conditional'
    - strict_data_scope == 'partial_real_seeded'
    - date fields parseable when non-null
    - ticker count positive integer when non-null
    """
    if not _SUMMARY_JSON.exists():
        pytest.skip(f"{_SUMMARY_JSON} not found — run scripts/run_pipeline.py --mode all")

    summary = json.loads(_SUMMARY_JSON.read_text())

    required_fields = [
        "phase_7_verdict",
        "strict_fixture_passed",
        "degraded_real_passed",
        "strict_real_passed",
        "production_strict_pipeline_passed",
        "strict_data_scope",
        "strict_ticker_count",
        "strict_first_valid_week",
        "strict_last_valid_week",
        "strict_micro_source_range",
        "strict_real_production_eligible",
        "strict_real_contract_grade",
    ]
    missing = [f for f in required_fields if f not in summary]
    assert not missing, f"Missing required fields in pipeline_mode_summary.json: {missing}"

    assert summary["phase_7_verdict"] == "pass_conditional", (
        f"phase_7_verdict must be 'pass_conditional', got {summary['phase_7_verdict']!r}"
    )
    assert summary["production_strict_pipeline_passed"] is False, (
        "production_strict_pipeline_passed must be False at this stage"
    )
    assert summary["strict_real_production_eligible"] is False, (
        "strict_real_production_eligible must be False (partial seeded data)"
    )
    assert summary["strict_real_contract_grade"] == "conditional", (
        f"strict_real_contract_grade must be 'conditional', got {summary['strict_real_contract_grade']!r}"
    )
    assert summary["strict_data_scope"] == "partial_real_seeded", (
        f"strict_data_scope must be 'partial_real_seeded', got {summary['strict_data_scope']!r}"
    )

    if summary["strict_first_valid_week"] is not None:
        pd.Timestamp(summary["strict_first_valid_week"])
    if summary["strict_last_valid_week"] is not None:
        pd.Timestamp(summary["strict_last_valid_week"])

    if summary["strict_ticker_count"] is not None:
        assert isinstance(summary["strict_ticker_count"], int) and summary["strict_ticker_count"] > 0, (
            f"strict_ticker_count must be a positive integer, got {summary['strict_ticker_count']!r}"
        )

    if _MANIFEST.exists():
        assert summary["strict_micro_source_range"] != "unavailable", (
            "strict_micro_source_range is 'unavailable' but seed_manifest.json exists"
        )
