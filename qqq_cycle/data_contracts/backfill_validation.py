"""Validation gates for controlled QQQ holdings backfill.

Inputs:
    Normalized holdings in canonical namespace and the price matrix namespace.
Outputs:
    Explicit strict/degraded validation booleans plus one machine reason code.
Time semantics:
    Validation uses the supplied normalized snapshot and namespace only. It
    does not infer freshness or publication timing from runtime state.
As-of semantics:
    Join coverage is evaluated against the supplied point-in-time price matrix
    namespace, not raw tickers or current listings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from qqq_cycle.data_contracts.instrument_namespace import NORMALIZATION_STATUSES


VALIDATION_REASONS = frozenset(
    {
        "validation_passed_strict",
        "validation_passed_degraded_only",
        "official_source_incomplete",
        "normalization_failure",
        "weight_sum_violation",
        "unresolved_weight_violation",
        "insufficient_join_coverage",
        "duplicate_instrument_id",
        "empty_holdings",
        "price_namespace_missing",
    }
)


@dataclass(frozen=True)
class BackfillValidationResult:
    weight_sum: float
    weight_sum_ok: bool
    join_coverage_weight: float
    join_coverage_ok: bool
    unresolved_weight_sum: float
    unresolved_weight_ok: bool
    strict_validation_ok: bool
    degraded_validation_ok: bool
    validation_reason: str


def _price_namespace_ids(price_matrix_namespace: Iterable[str] | pd.DataFrame | None) -> set[str] | None:
    if price_matrix_namespace is None:
        return None
    if isinstance(price_matrix_namespace, pd.DataFrame):
        if price_matrix_namespace.empty or "instrument_id" not in price_matrix_namespace.columns:
            return None
        return {str(value) for value in price_matrix_namespace["instrument_id"].dropna()}
    ids = {str(value) for value in price_matrix_namespace}
    return ids or None


def _result(
    *,
    weight_sum: float,
    strict_weight_sum_ok: bool,
    degraded_weight_sum_ok: bool,
    join_coverage_weight: float,
    strict_join_ok: bool,
    degraded_join_ok: bool,
    unresolved_weight_sum: float,
    unresolved_weight_ok: bool,
    validation_reason: str,
) -> BackfillValidationResult:
    if validation_reason not in VALIDATION_REASONS:
        raise ValueError(f"unknown validation reason: {validation_reason}")
    strict_ok = strict_weight_sum_ok and strict_join_ok and unresolved_weight_ok and validation_reason in {
        "validation_passed_strict",
        "validation_passed_degraded_only",
    }
    degraded_ok = degraded_weight_sum_ok and degraded_join_ok and unresolved_weight_ok and validation_reason in {
        "validation_passed_strict",
        "validation_passed_degraded_only",
    }
    return BackfillValidationResult(
        weight_sum=weight_sum,
        weight_sum_ok=strict_weight_sum_ok,
        join_coverage_weight=join_coverage_weight,
        join_coverage_ok=strict_join_ok,
        unresolved_weight_sum=unresolved_weight_sum,
        unresolved_weight_ok=unresolved_weight_ok,
        strict_validation_ok=strict_ok,
        degraded_validation_ok=degraded_ok,
        validation_reason=validation_reason,
    )


def validate_normalized_holdings(
    normalized_holdings: pd.DataFrame,
    price_matrix_namespace: Iterable[str] | pd.DataFrame | None,
    strict_weight_sum_min: float = 0.99,
    strict_weight_sum_max: float = 1.01,
    strict_join_coverage_min: float = 0.99,
    strict_unresolved_weight_max: float = 0.01,
    degraded_weight_sum_min: float = 0.98,
    degraded_weight_sum_max: float = 1.02,
    degraded_join_coverage_min: float = 0.95,
    degraded_unresolved_weight_max: float = 0.01,
) -> BackfillValidationResult:
    """Apply hard validation gates to normalized controlled backfill holdings."""

    empty_result = {
        "weight_sum": 0.0,
        "strict_weight_sum_ok": False,
        "degraded_weight_sum_ok": False,
        "join_coverage_weight": 0.0,
        "strict_join_ok": False,
        "degraded_join_ok": False,
        "unresolved_weight_sum": 0.0,
        "unresolved_weight_ok": False,
    }
    if normalized_holdings.empty:
        return _result(**empty_result, validation_reason="empty_holdings")

    required = {"instrument_id", "normalized_weight", "normalization_status"}
    if missing := required.difference(normalized_holdings.columns):
        return _result(**empty_result, validation_reason="normalization_failure")

    ids = _price_namespace_ids(price_matrix_namespace)
    if ids is None:
        return _result(**empty_result, validation_reason="price_namespace_missing")

    frame = normalized_holdings.copy()
    frame["normalized_weight"] = pd.to_numeric(frame["normalized_weight"], errors="coerce")
    if frame["normalized_weight"].isna().any():
        return _result(**empty_result, validation_reason="normalization_failure")
    if not set(frame["normalization_status"]).issubset(NORMALIZATION_STATUSES):
        return _result(**empty_result, validation_reason="normalization_failure")

    joined_required = (
        frame["price_join_required"].astype(bool)
        if "price_join_required" in frame.columns
        else pd.Series(True, index=frame.index)
    )
    resolved = frame["normalization_status"] != "unresolved"
    duplicate_scope = frame.loc[joined_required & resolved & frame["instrument_id"].astype(str).ne("")]
    if duplicate_scope["instrument_id"].duplicated().any():
        allowed = (
            duplicate_scope.get("aggregation_allowed", pd.Series(False, index=duplicate_scope.index))
            .fillna(False)
            .astype(bool)
        )
        if not bool(allowed.all()):
            metrics = _metrics(
                frame,
                ids,
                strict_weight_sum_min,
                strict_weight_sum_max,
                strict_join_coverage_min,
                strict_unresolved_weight_max,
                degraded_weight_sum_min,
                degraded_weight_sum_max,
                degraded_join_coverage_min,
                degraded_unresolved_weight_max,
            )
            return _result(**metrics, validation_reason="duplicate_instrument_id")

    metrics = _metrics(
        frame,
        ids,
        strict_weight_sum_min,
        strict_weight_sum_max,
        strict_join_coverage_min,
        strict_unresolved_weight_max,
        degraded_weight_sum_min,
        degraded_weight_sum_max,
        degraded_join_coverage_min,
        degraded_unresolved_weight_max,
    )
    if not metrics["degraded_weight_sum_ok"]:
        return _result(**metrics, validation_reason="weight_sum_violation")
    if not metrics["unresolved_weight_ok"]:
        return _result(**metrics, validation_reason="unresolved_weight_violation")
    if not metrics["degraded_join_ok"]:
        return _result(**metrics, validation_reason="insufficient_join_coverage")
    if metrics["strict_weight_sum_ok"] and metrics["strict_join_ok"]:
        return _result(**metrics, validation_reason="validation_passed_strict")
    return _result(**metrics, validation_reason="validation_passed_degraded_only")


def _metrics(
    frame: pd.DataFrame,
    price_ids: set[str],
    strict_weight_sum_min: float,
    strict_weight_sum_max: float,
    strict_join_coverage_min: float,
    strict_unresolved_weight_max: float,
    degraded_weight_sum_min: float,
    degraded_weight_sum_max: float,
    degraded_join_coverage_min: float,
    degraded_unresolved_weight_max: float,
) -> dict[str, float | bool]:
    weights = frame["normalized_weight"].astype(float)
    weight_sum = float(weights.sum())
    unresolved_weight_sum = float(weights[frame["normalization_status"] == "unresolved"].sum())
    join_required = (
        frame["price_join_required"].astype(bool)
        if "price_join_required" in frame.columns
        else pd.Series(True, index=frame.index)
    )
    join_denominator = float(weights[join_required].sum())
    if join_denominator == 0.0:
        join_coverage_weight = 1.0
    else:
        joined = join_required & frame["instrument_id"].astype(str).isin(price_ids)
        join_coverage_weight = float(weights[joined].sum() / join_denominator)
    return {
        "weight_sum": weight_sum,
        "strict_weight_sum_ok": strict_weight_sum_min <= weight_sum <= strict_weight_sum_max,
        "degraded_weight_sum_ok": degraded_weight_sum_min <= weight_sum <= degraded_weight_sum_max,
        "join_coverage_weight": join_coverage_weight,
        "strict_join_ok": join_coverage_weight >= strict_join_coverage_min,
        "degraded_join_ok": join_coverage_weight >= degraded_join_coverage_min,
        "unresolved_weight_sum": unresolved_weight_sum,
        "unresolved_weight_ok": unresolved_weight_sum
        <= min(strict_unresolved_weight_max, degraded_unresolved_weight_max),
    }
