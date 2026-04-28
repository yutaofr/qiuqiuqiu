"""Phase 8 production strict blocker registry tests."""

from __future__ import annotations

import json
from pathlib import Path


_REGISTRY_JSON = Path("outputs/production_strict_blockers.json")
_ACCEPTANCE_MD = Path("outputs/production_input_chain_acceptance.md")
_SUMMARY_JSON = Path("outputs/pipeline/pipeline_mode_summary.json")


def test_production_strict_blocker_registry_is_machine_readable() -> None:
    """Registry schema is parseable and preserves the Phase 9 approval status."""
    registry = json.loads(_REGISTRY_JSON.read_text())

    assert registry["schema_version"] == "1.0"
    assert registry["phase"] == "phase_9"
    assert isinstance(registry["generated_at"], str)
    assert isinstance(registry["closed"], list)
    assert isinstance(registry["open"], list)

    summary = registry["summary"]
    assert summary["total_blockers"] == len(registry["closed"]) + len(registry["open"])
    assert summary["closed"] == len(registry["closed"])
    assert summary["open"] == len(registry["open"])
    assert summary["production_strict_pipeline_passed"] is True
    assert summary["phase_9_verdict"] == "production_strict_approved"
    assert summary["production_strict_epoch_start"] is not None

    closed_ids = {item["id"] for item in registry["closed"]}
    open_ids = {item["id"] for item in registry["open"]}
    assert {
        "pit_source_asof_semantics_documented",
        "pit_chained_compounding_verified",
        "pit_no_lookahead_cutoff_verified",
        "constituent_semantics_documented",
        "survivor_bias_constituent_behavior_tested",
        "weight_sum_validation_available",
        "missing_weight_no_silent_fill_verified",
        "weight_boundary_behavior_verified",
    } <= closed_ids
    assert open_ids == set()
    assert {
        "ledger_pit_engine_enabled",
        "symbol_identity_bridge_enabled",
        "production_strict_epoch_machine_derived",
    } <= closed_ids


def test_production_input_chain_acceptance_matches_registry() -> None:
    """Acceptance markdown mirrors blocker IDs and fixed Phase 8 verdicts."""
    registry = json.loads(_REGISTRY_JSON.read_text())
    markdown = _ACCEPTANCE_MD.read_text()

    assert "phase_9_verdict = production_strict_approved" in markdown
    assert "production_strict_pipeline_passed = true" in markdown
    assert "production_strict_path = approved" in markdown
    assert "Remaining Production Blockers" in markdown
    assert "- none" in markdown

    for item in registry["open"] + registry["closed"]:
        assert item["id"] in markdown

    summary = registry["summary"]
    assert f"total_blockers = {summary['total_blockers']}" in markdown
    assert f"closed = {summary['closed']}" in markdown
    assert f"open = {summary['open']}" in markdown


def test_pipeline_summary_phase8_fields_match_registry() -> None:
    """Pipeline summary exposes Phase 8 registry counts without changing verdict."""
    registry = json.loads(_REGISTRY_JSON.read_text())
    summary = json.loads(_SUMMARY_JSON.read_text())
    registry_summary = registry["summary"]

    assert summary["production_strict_pipeline_passed"] is True
    assert summary["phase_9_verdict"] == registry_summary["phase_9_verdict"]
    assert summary["production_strict_epoch_start"] == registry_summary["production_strict_epoch_start"]
    assert summary["production_strict_path"] == "approved"
    assert summary["phase_8_blocker_count_open"] == registry_summary["open"]
    assert summary["phase_8_blocker_count_closed"] == registry_summary["closed"]
    assert summary["phase_8_blocker_registry"] == str(_REGISTRY_JSON)
