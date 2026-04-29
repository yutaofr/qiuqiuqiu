from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qqq_cycle.live.controlled_backfill_contracts import resolve_live_contracts_for_week
from scripts.run_live_pipeline import _controlled_backfill_mode_for_week


WEEK = "2026-04-24"


def _write_legacy_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "prices").mkdir()
    pd.DataFrame(
        [
            {"trade_date": WEEK, "ticker": "AAPL", "asof_timestamp": f"{WEEK}T16:00:00"},
        ]
    ).to_csv(cache_dir / "constituents.csv", index=False)
    pd.DataFrame(
        [
            {"trade_date": WEEK, "ticker": "AAPL", "weight": 1.0, "asof_timestamp": f"{WEEK}T16:00:00"},
        ]
    ).to_csv(cache_dir / "weights.csv", index=False)
    pd.DataFrame(
        [
            {"trade_date": date.strftime("%Y-%m-%d"), "raw_close": 100.0 + idx, "asof_timestamp": f"{date.strftime('%Y-%m-%d')}T16:00:00"}
            for idx, date in enumerate(pd.bdate_range(end=WEEK, periods=60))
        ]
    ).to_csv(cache_dir / "prices" / "AAPL.csv", index=False)


def _controlled(mode: str) -> dict:
    return {
        "week_end": WEEK,
        "asset": "QQQ",
        "backfill_mode": mode,
        "publication_proof_class": "direct_http_capture_at_or_before_sla",
        "strict_eligible": mode == "strict_recovery",
        "revision_reason": "controlled_backfill_without_pit_proof",
        "validation_reason": "validation_passed_strict",
        "decision_reason": "degraded_backfill_without_pit_proof",
        "content_sha256": "a" * 64,
        "normalized_sha256": "b" * 64,
        "created_at_utc": "2026-04-29T00:00:00Z",
    }


def _write_valid_strict_store(store_root: Path, *, with_weekly_h_t: bool = True) -> None:
    const_dir = store_root / "strict" / "constituents"
    weight_dir = store_root / "strict" / "weights"
    const_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"instrument_id": "AAPL", "canonical_symbol": "AAPL", "asset_class": "equity", "normalization_status": "resolved_by_exact_symbol_map"}]
    ).to_csv(const_dir / f"qqq_constituents_{WEEK}.csv", index=False)
    pd.DataFrame([{"instrument_id": "AAPL", "normalized_weight": 1.0}]).to_csv(
        weight_dir / f"qqq_weights_{WEEK}.csv", index=False
    )
    if with_weekly_h_t:
        h_dir = store_root / "strict" / "weekly_h_t"
        h_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"week_end": WEEK, "h_t": 0.7}]).to_csv(
            h_dir / f"qqq_weekly_h_t_{WEEK}.csv", index=False
        )


def test_controlled_block_with_legacy_cache_does_not_build_strict_contracts(tmp_path: Path) -> None:
    _write_legacy_cache(tmp_path / "cache" / "micro")

    result = resolve_live_contracts_for_week(
        week_end=WEEK,
        asset="QQQ",
        controlled_backfill_result=_controlled("block"),
        store_root=tmp_path / "stores",
        legacy_cache_micro_dir=tmp_path / "cache" / "micro",
        production_ledger_dir=tmp_path / "ledgers",
    )

    assert result.contracts is None
    assert result.backfill_mode == "block"
    assert result.contract_source == "controlled_block"
    assert result.strict_gate_passed is False


def test_degraded_backfill_with_legacy_cache_does_not_use_cache_micro(tmp_path: Path) -> None:
    _write_legacy_cache(tmp_path / "cache" / "micro")

    result = resolve_live_contracts_for_week(
        week_end=WEEK,
        asset="QQQ",
        controlled_backfill_result=_controlled("degraded_backfill"),
        store_root=tmp_path / "stores",
        legacy_cache_micro_dir=tmp_path / "cache" / "micro",
        production_ledger_dir=tmp_path / "ledgers",
    )

    assert result.contracts is None
    assert result.backfill_mode == "degraded_backfill"
    assert result.contract_source == "stores_backfill"
    assert result.strict_gate_passed is False


def test_strict_recovery_reads_only_stores_strict_when_gate_passes(tmp_path: Path) -> None:
    _write_legacy_cache(tmp_path / "cache" / "micro")
    _write_valid_strict_store(tmp_path / "stores")

    result = resolve_live_contracts_for_week(
        week_end=WEEK,
        asset="QQQ",
        controlled_backfill_result=_controlled("strict_recovery"),
        store_root=tmp_path / "stores",
        legacy_cache_micro_dir=tmp_path / "cache" / "micro",
        production_ledger_dir=tmp_path / "ledgers",
    )

    assert result.contracts is not None
    assert result.contracts.weekly_h_t is not None
    assert result.backfill_mode == "strict_recovery"
    assert result.contract_source == "stores_strict"
    assert result.strict_gate_passed is True


def test_strict_recovery_invalid_strict_store_never_falls_back_to_cache(tmp_path: Path) -> None:
    _write_legacy_cache(tmp_path / "cache" / "micro")

    result = resolve_live_contracts_for_week(
        week_end=WEEK,
        asset="QQQ",
        controlled_backfill_result=_controlled("strict_recovery"),
        store_root=tmp_path / "stores",
        legacy_cache_micro_dir=tmp_path / "cache" / "micro",
        production_ledger_dir=tmp_path / "ledgers",
    )

    assert result.contracts is None
    assert result.backfill_mode == "block"
    assert result.contract_source == "controlled_block"
    assert result.strict_gate_passed is False
    assert result.reason.startswith("strict_store_gate_failed")


def test_non_controlled_week_can_use_legacy_cache_micro(tmp_path: Path) -> None:
    _write_legacy_cache(tmp_path / "cache" / "micro")

    result = resolve_live_contracts_for_week(
        week_end=WEEK,
        asset="QQQ",
        controlled_backfill_result=None,
        store_root=tmp_path / "stores",
        legacy_cache_micro_dir=tmp_path / "cache" / "micro",
        production_ledger_dir=tmp_path / "ledgers",
    )

    assert result.contracts is not None
    assert result.backfill_mode is None
    assert result.contract_source == "legacy_cache_micro"


def test_run_live_pipeline_default_controlled_lookup_has_priority_over_cache(tmp_path: Path) -> None:
    _write_legacy_cache(tmp_path / "cache" / "micro")
    phase14 = tmp_path / "outputs" / "phase14"
    phase14.mkdir(parents=True)
    (phase14 / f"controlled_backfill_result_qqq_{WEEK}.json").write_text(
        json.dumps(_controlled("degraded_backfill")),
        encoding="utf-8",
    )

    assert _controlled_backfill_mode_for_week(WEEK, phase14) == "degraded_backfill"
