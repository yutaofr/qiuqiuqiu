from __future__ import annotations

import pandas as pd

from qqq_cycle.ops.backfill_ingest import (
    BackfillDecision,
    evaluate_strict_store_gate,
    write_backfill_stores,
)


WEEK = "2026-04-24"


def normalized() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "instrument_id": "I1",
                "canonical_symbol": "AAA",
                "asset_class": "equity",
                "normalization_status": "resolved_by_exact_symbol_map",
                "normalized_weight": 1.0,
            }
        ]
    )


def decision(scheme: str) -> BackfillDecision:
    return BackfillDecision(
        scheme=scheme,  # type: ignore[arg-type]
        reason=(
            "strict_recovery_verified_pit_availability"
            if scheme == "strict_recovery"
            else "degraded_backfill_without_pit_proof"
            if scheme == "degraded_backfill"
            else "block_missing_or_invalid_proof_and_validation"
        ),
        strict_eligible=scheme == "strict_recovery",
        strict_validation_ok=scheme == "strict_recovery",
        degraded_validation_ok=scheme != "block",
    )


def test_strict_recovery_writes_strict_only(tmp_path) -> None:
    result = write_backfill_stores(
        normalized_holdings=normalized(),
        decision=decision("strict_recovery"),
        week_end=WEEK,
        store_root=tmp_path,
    )

    assert result.strict_constituents_path is not None
    assert result.strict_weights_path is not None
    assert result.backfill_constituents_path is None
    assert result.backfill_weights_path is None
    assert not (tmp_path / "backfill").exists()


def test_degraded_backfill_writes_backfill_only(tmp_path) -> None:
    result = write_backfill_stores(
        normalized_holdings=normalized(),
        decision=decision("degraded_backfill"),
        week_end=WEEK,
        store_root=tmp_path,
    )

    assert result.strict_constituents_path is None
    assert result.strict_weights_path is None
    assert result.backfill_constituents_path is not None
    assert result.backfill_weights_path is not None
    assert not (tmp_path / "strict").exists()


def test_block_writes_neither_store(tmp_path) -> None:
    result = write_backfill_stores(
        normalized_holdings=normalized(),
        decision=decision("block"),
        week_end=WEEK,
        store_root=tmp_path,
    )

    assert result == result.__class__(None, None, None, None)
    assert not (tmp_path / "strict").exists()
    assert not (tmp_path / "backfill").exists()


def test_strict_gate_cannot_read_backfill_path(tmp_path) -> None:
    write_backfill_stores(
        normalized_holdings=normalized(),
        decision=decision("degraded_backfill"),
        week_end=WEEK,
        store_root=tmp_path,
    )

    gate = evaluate_strict_store_gate(week_end=WEEK, store_root=tmp_path)

    assert gate.passed is False
    assert gate.reason == "strict_store_missing"


def test_strict_gate_cannot_pass_on_backfill_only_data(tmp_path) -> None:
    write_backfill_stores(
        normalized_holdings=normalized(),
        decision=decision("degraded_backfill"),
        week_end=WEEK,
        store_root=tmp_path,
    )

    assert evaluate_strict_store_gate(week_end=WEEK, store_root=tmp_path).passed is False


def test_strict_gate_cannot_pass_from_normalized_file_only(tmp_path) -> None:
    normalized_path = tmp_path / "normalized" / f"qqq_holdings_{WEEK}_normalized.csv"
    normalized_path.parent.mkdir()
    normalized().to_csv(normalized_path, index=False)

    gate = evaluate_strict_store_gate(week_end=WEEK, store_root=tmp_path)

    assert gate.passed is False
    assert gate.reason == "strict_store_missing"


def test_strict_gate_passes_on_valid_strict_store_contents(tmp_path) -> None:
    write_backfill_stores(
        normalized_holdings=normalized(),
        decision=decision("strict_recovery"),
        week_end=WEEK,
        store_root=tmp_path,
    )

    gate = evaluate_strict_store_gate(week_end=WEEK, store_root=tmp_path)

    assert gate.passed is True
    assert gate.reason == "strict_store_valid"
