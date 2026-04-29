"""Controlled backfill ingest decision and store separation.

Inputs:
    Publication proof evaluation, normalized holdings validation, and canonical
    normalized holdings rows.
Outputs:
    A scheme from exactly strict_recovery, degraded_backfill, or block plus
    physically separated strict/backfill store writes.
Time semantics:
    The decision consumes already-evaluated point-in-time proof and validation
    results. It does not infer publication timing from files or runtime.
As-of semantics:
    Strict stores are populated only by strict_recovery decisions. Degraded
    backfills cannot contaminate the strict namespace.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import json
import pandas as pd

from qqq_cycle.data_contracts.backfill_validation import BackfillValidationResult
from qqq_cycle.data_contracts.publication_proof import PublicationProof


BackfillScheme = Literal["strict_recovery", "degraded_backfill", "block"]

DECISION_REASONS = frozenset(
    {
        "strict_recovery_verified_pit_availability",
        "degraded_backfill_without_pit_proof",
        "degraded_backfill_validation_only",
        "block_missing_or_invalid_proof_and_validation",
        "block_official_source_incomplete",
        "block_normalization_failure",
        "block_weight_sum_violation",
        "block_unresolved_weight_violation",
        "block_insufficient_join_coverage",
    }
)

REVISION_REASONS = frozenset(
    {
        "controlled_backfill_with_verified_pit_availability",
        "controlled_backfill_without_pit_proof",
        "namespace_normalization_failure",
        "weight_sum_violation",
        "insufficient_join_coverage",
    }
)


@dataclass(frozen=True)
class BackfillDecision:
    scheme: BackfillScheme
    reason: str
    strict_eligible: bool
    strict_validation_ok: bool
    degraded_validation_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlledBackfillResult:
    week_end: str
    asset: str
    backfill_mode: BackfillScheme
    publication_proof_class: str
    strict_eligible: bool
    revision_reason: str
    validation_reason: str
    decision_reason: str
    content_sha256: str
    normalized_sha256: str
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StoreWriteResult:
    strict_constituents_path: Path | None
    strict_weights_path: Path | None
    backfill_constituents_path: Path | None
    backfill_weights_path: Path | None


@dataclass(frozen=True)
class StrictGateResult:
    passed: bool
    reason: str
    constituents_path: Path
    weights_path: Path


def decide_backfill_scheme(
    proof_result: PublicationProof,
    validation_result: BackfillValidationResult,
) -> BackfillDecision:
    """Return exactly one controlled backfill scheme from proof and validation."""

    proof_ok = bool(proof_result.strict_eligible)
    strict_ok = bool(validation_result.strict_validation_ok)
    degraded_ok = bool(validation_result.degraded_validation_ok)

    if proof_ok and strict_ok:
        return _decision(
            "strict_recovery",
            "strict_recovery_verified_pit_availability",
            proof_ok,
            strict_ok,
            degraded_ok,
        )
    if degraded_ok:
        return _decision(
            "degraded_backfill",
            "degraded_backfill_validation_only" if proof_ok else "degraded_backfill_without_pit_proof",
            proof_ok,
            strict_ok,
            degraded_ok,
        )
    return _decision(
        "block",
        _block_reason(validation_result.validation_reason, proof_ok),
        proof_ok,
        strict_ok,
        degraded_ok,
    )


def write_backfill_stores(
    *,
    normalized_holdings: pd.DataFrame,
    decision: BackfillDecision,
    week_end: str,
    asset: str = "QQQ",
    store_root: str | Path = "stores",
) -> StoreWriteResult:
    """Write strict or backfill stores according to the selected scheme."""

    root = Path(store_root)
    asset_lc = asset.lower()
    strict_constituents = root / "strict" / "constituents" / f"{asset_lc}_constituents_{week_end}.csv"
    strict_weights = root / "strict" / "weights" / f"{asset_lc}_weights_{week_end}.csv"
    backfill_constituents = root / "backfill" / "constituents" / f"{asset_lc}_constituents_{week_end}.csv"
    backfill_weights = root / "backfill" / "weights" / f"{asset_lc}_weights_{week_end}.csv"

    if decision.scheme == "block":
        return StoreWriteResult(None, None, None, None)

    target_constituents, target_weights = (
        (strict_constituents, strict_weights)
        if decision.scheme == "strict_recovery"
        else (backfill_constituents, backfill_weights)
    )
    target_constituents.parent.mkdir(parents=True, exist_ok=True)
    target_weights.parent.mkdir(parents=True, exist_ok=True)

    constituents_cols = [
        col
        for col in ("instrument_id", "canonical_symbol", "asset_class", "normalization_status")
        if col in normalized_holdings.columns
    ]
    if "instrument_id" not in constituents_cols:
        raise ValueError("normalized holdings missing instrument_id for store write")
    weights_cols = [col for col in ("instrument_id", "normalized_weight") if col in normalized_holdings.columns]
    if set(weights_cols) != {"instrument_id", "normalized_weight"}:
        raise ValueError("normalized holdings missing weight columns for store write")

    normalized_holdings.loc[:, constituents_cols].drop_duplicates().to_csv(
        target_constituents, index=False
    )
    normalized_holdings.loc[:, weights_cols].to_csv(target_weights, index=False)

    return StoreWriteResult(
        strict_constituents if decision.scheme == "strict_recovery" else None,
        strict_weights if decision.scheme == "strict_recovery" else None,
        backfill_constituents if decision.scheme == "degraded_backfill" else None,
        backfill_weights if decision.scheme == "degraded_backfill" else None,
    )


def evaluate_strict_store_gate(
    *,
    week_end: str,
    asset: str = "QQQ",
    store_root: str | Path = "stores",
) -> StrictGateResult:
    """Evaluate strict legality using only strict store namespace contents."""

    root = Path(store_root)
    asset_lc = asset.lower()
    constituents_path = root / "strict" / "constituents" / f"{asset_lc}_constituents_{week_end}.csv"
    weights_path = root / "strict" / "weights" / f"{asset_lc}_weights_{week_end}.csv"
    if not constituents_path.exists() or not weights_path.exists():
        return StrictGateResult(False, "strict_store_missing", constituents_path, weights_path)

    constituents = pd.read_csv(constituents_path)
    weights = pd.read_csv(weights_path)
    if constituents.empty or weights.empty:
        return StrictGateResult(False, "strict_store_empty", constituents_path, weights_path)
    if "instrument_id" not in constituents.columns or not {"instrument_id", "normalized_weight"}.issubset(weights.columns):
        return StrictGateResult(False, "strict_store_schema_invalid", constituents_path, weights_path)

    constituent_ids = set(constituents["instrument_id"].astype(str))
    weight_ids = set(weights["instrument_id"].astype(str))
    if not weight_ids.issubset(constituent_ids):
        return StrictGateResult(False, "strict_store_namespace_mismatch", constituents_path, weights_path)

    weight_sum = float(pd.to_numeric(weights["normalized_weight"], errors="coerce").sum())
    if not 0.99 <= weight_sum <= 1.01:
        return StrictGateResult(False, "strict_store_weight_sum_violation", constituents_path, weights_path)

    return StrictGateResult(True, "strict_store_valid", constituents_path, weights_path)


def controlled_backfill_result_from_decision(
    *,
    week_end: str,
    asset: str,
    proof_result: PublicationProof,
    validation_result: BackfillValidationResult,
    decision: BackfillDecision,
    normalized_holdings: pd.DataFrame,
    created_at_utc: str | None = None,
) -> ControlledBackfillResult:
    """Build an auditable result artifact from the decision engine output."""

    created = created_at_utc or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    revision_reason = _revision_reason(decision, validation_result)
    return ControlledBackfillResult(
        week_end=week_end,
        asset=asset,
        backfill_mode=decision.scheme,
        publication_proof_class=proof_result.evidence_class,
        strict_eligible=bool(proof_result.strict_eligible),
        revision_reason=revision_reason,
        validation_reason=validation_result.validation_reason,
        decision_reason=decision.reason,
        content_sha256=proof_result.content_sha256,
        normalized_sha256=_normalized_sha256(normalized_holdings),
        created_at_utc=created,
    )


def write_controlled_backfill_result(
    result: ControlledBackfillResult,
    *,
    output_dir: str | Path = "outputs/phase14",
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"controlled_backfill_result_{result.asset.lower()}_{result.week_end}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_controlled_backfill_result(
    *,
    week_end: str,
    asset: str = "QQQ",
    output_dir: str | Path = "outputs/phase14",
) -> dict[str, Any] | None:
    """Load the controlled backfill result for a week if it exists."""

    path = Path(output_dir) / f"controlled_backfill_result_{asset.lower()}_{week_end}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_latest_controlled_backfill_result(
    *,
    asset: str = "QQQ",
    output_dir: str | Path = "outputs/phase14",
) -> dict[str, Any] | None:
    """Load the latest controlled backfill result by filename order."""

    matches = sorted(Path(output_dir).glob(f"controlled_backfill_result_{asset.lower()}_*.json"))
    if not matches:
        return None
    return json.loads(matches[-1].read_text(encoding="utf-8"))


def _decision(
    scheme: BackfillScheme,
    reason: str,
    strict_eligible: bool,
    strict_validation_ok: bool,
    degraded_validation_ok: bool,
) -> BackfillDecision:
    if reason not in DECISION_REASONS:
        raise ValueError(f"unknown decision reason: {reason}")
    return BackfillDecision(
        scheme=scheme,
        reason=reason,
        strict_eligible=strict_eligible,
        strict_validation_ok=strict_validation_ok,
        degraded_validation_ok=degraded_validation_ok,
    )


def _block_reason(validation_reason: str, proof_ok: bool) -> str:
    if validation_reason == "official_source_incomplete" or validation_reason == "empty_holdings":
        return "block_official_source_incomplete"
    if validation_reason in {"normalization_failure", "duplicate_instrument_id", "price_namespace_missing"}:
        return "block_normalization_failure"
    if validation_reason == "weight_sum_violation":
        return "block_weight_sum_violation"
    if validation_reason == "unresolved_weight_violation":
        return "block_unresolved_weight_violation"
    if validation_reason == "insufficient_join_coverage":
        return "block_insufficient_join_coverage"
    return "block_missing_or_invalid_proof_and_validation"


def _revision_reason(
    decision: BackfillDecision,
    validation_result: BackfillValidationResult,
) -> str:
    if decision.scheme == "strict_recovery":
        return "controlled_backfill_with_verified_pit_availability"
    if decision.scheme == "degraded_backfill":
        return "controlled_backfill_without_pit_proof"
    if validation_result.validation_reason in {"normalization_failure", "duplicate_instrument_id", "price_namespace_missing"}:
        return "namespace_normalization_failure"
    if validation_result.validation_reason in {"weight_sum_violation", "unresolved_weight_violation"}:
        return "weight_sum_violation"
    if validation_result.validation_reason == "insufficient_join_coverage":
        return "insufficient_join_coverage"
    return "namespace_normalization_failure"


def _normalized_sha256(normalized_holdings: pd.DataFrame) -> str:
    stable = normalized_holdings.fillna("").astype(str).sort_index(axis=1).to_csv(index=False)
    return sha256(stable.encode("utf-8")).hexdigest()
