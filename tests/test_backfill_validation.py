from __future__ import annotations

import pandas as pd
import pytest

from qqq_cycle.data_contracts.backfill_validation import validate_normalized_holdings


def holdings(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "instrument_id": row["instrument_id"],
                "normalized_weight": row["weight"],
                "normalization_status": row.get("status", "resolved_by_exact_symbol_map"),
                "price_join_required": row.get("price_join_required", True),
                **({"aggregation_allowed": row["aggregation_allowed"]} if "aggregation_allowed" in row else {}),
            }
            for row in rows
        ]
    )


def test_strict_validation_ok() -> None:
    result = validate_normalized_holdings(
        holdings([{"instrument_id": "I1", "weight": 0.995}, {"instrument_id": "I2", "weight": 0.005}]),
        {"I1"},
    )

    assert result.weight_sum == pytest.approx(1.0)
    assert result.join_coverage_weight == pytest.approx(0.995)
    assert result.strict_validation_ok is True
    assert result.degraded_validation_ok is True
    assert result.validation_reason == "validation_passed_strict"


def test_degraded_only_validation_ok() -> None:
    result = validate_normalized_holdings(
        holdings([{"instrument_id": "I1", "weight": 0.97}, {"instrument_id": "I2", "weight": 0.015}]),
        {"I1"},
    )

    assert result.weight_sum == pytest.approx(0.985)
    assert result.join_coverage_weight == pytest.approx(0.97 / 0.985)
    assert result.strict_validation_ok is False
    assert result.degraded_validation_ok is True
    assert result.validation_reason == "validation_passed_degraded_only"


def test_weight_sum_below_degraded_blocks() -> None:
    result = validate_normalized_holdings(holdings([{"instrument_id": "I1", "weight": 0.97}]), {"I1"})

    assert result.strict_validation_ok is False
    assert result.degraded_validation_ok is False
    assert result.validation_reason == "weight_sum_violation"


def test_join_coverage_below_degraded_blocks() -> None:
    result = validate_normalized_holdings(
        holdings([{"instrument_id": "I1", "weight": 0.94}, {"instrument_id": "I2", "weight": 0.06}]),
        {"I1"},
    )

    assert result.join_coverage_weight == pytest.approx(0.94)
    assert result.degraded_validation_ok is False
    assert result.validation_reason == "insufficient_join_coverage"


def test_unresolved_weight_above_one_percent_blocks() -> None:
    result = validate_normalized_holdings(
        holdings(
            [
                {"instrument_id": "I1", "weight": 0.989},
                {"instrument_id": "", "weight": 0.011, "status": "unresolved"},
            ]
        ),
        {"I1"},
    )

    assert result.unresolved_weight_sum == pytest.approx(0.011)
    assert result.validation_reason == "unresolved_weight_violation"
    assert result.degraded_validation_ok is False


def test_empty_holdings_blocks() -> None:
    result = validate_normalized_holdings(pd.DataFrame(), {"I1"})

    assert result.validation_reason == "empty_holdings"
    assert result.strict_validation_ok is False


def test_duplicate_instrument_id_without_allowed_aggregation_blocks() -> None:
    result = validate_normalized_holdings(
        holdings([{"instrument_id": "I1", "weight": 0.5}, {"instrument_id": "I1", "weight": 0.5}]),
        {"I1"},
    )

    assert result.validation_reason == "duplicate_instrument_id"
    assert result.degraded_validation_ok is False


def test_duplicate_instrument_id_with_allowed_aggregation_passes() -> None:
    result = validate_normalized_holdings(
        holdings(
            [
                {"instrument_id": "I1", "weight": 0.5, "aggregation_allowed": True},
                {"instrument_id": "I1", "weight": 0.5, "aggregation_allowed": True},
            ]
        ),
        {"I1"},
    )

    assert result.strict_validation_ok is True


def test_missing_price_namespace_blocks() -> None:
    result = validate_normalized_holdings(holdings([{"instrument_id": "I1", "weight": 1.0}]), None)

    assert result.validation_reason == "price_namespace_missing"
    assert result.strict_validation_ok is False
