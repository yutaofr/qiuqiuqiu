"""Phase 15 portfolio policy loading and validation.

Inputs:
    Policy configuration stored as a JSON-compatible YAML document.
Outputs:
    Immutable portfolio policy objects with validated bucket and weight rules.
Time semantics:
    Static configuration only; no market timestamps are synthesized here.
As-of semantics:
    The loaded policy is a frozen contract applied by downstream weekly logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


EPSILON = 1e-9


@dataclass(frozen=True)
class UniverseAsset:
    symbol: str
    asset_class: str
    min_weight: float
    max_weight: float
    price_required: bool


@dataclass(frozen=True)
class PortfolioConstraints:
    gross_exposure_max: float
    net_exposure_min: float
    net_exposure_max: float
    single_asset_max: float
    turnover_threshold: float
    rebalance_frequency: str
    leverage_allowed: bool
    shorting_allowed: bool


@dataclass(frozen=True)
class RhoBucket:
    name: str
    lower: float
    upper: float
    lower_inclusive: bool
    upper_inclusive: bool

    def contains(self, value: float) -> bool:
        lower_ok = value >= self.lower if self.lower_inclusive else value > self.lower
        upper_ok = value <= self.upper if self.upper_inclusive else value < self.upper
        return lower_ok and upper_ok


@dataclass(frozen=True)
class PortfolioPolicy:
    policy_id: str
    policy_version: int
    paper_only: bool
    broker_submission_allowed: bool
    base_currency: str
    universe: tuple[UniverseAsset, ...]
    constraints: PortfolioConstraints
    rho_buckets: tuple[RhoBucket, ...]
    state_policy: Mapping[str, Mapping[str, Mapping[str, float]]]
    execution_model: Mapping[str, Any]
    known_limitations: tuple[str, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(asset.symbol for asset in self.universe)

    def default_state_policy(self) -> Mapping[str, Mapping[str, float]]:
        return self.state_policy["strict_default"]

    def locate_rho_bucket(self, rho_t: float) -> str:
        for bucket in self.rho_buckets:
            if bucket.contains(rho_t):
                return bucket.name
        raise ValueError(f"rho_t={rho_t} does not fall into any configured bucket")


def _read_policy_mapping(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"policy file {path} must use JSON-compatible YAML syntax for deterministic loading"
        ) from exc


def _validate_bool_invariants(mapping: Mapping[str, Any]) -> None:
    if mapping.get("paper_only") is not True:
        raise ValueError("portfolio policy must set paper_only=true")
    if mapping.get("broker_submission_allowed") is not False:
        raise ValueError("portfolio policy must set broker_submission_allowed=false")


def _build_universe(items: list[Mapping[str, Any]]) -> tuple[UniverseAsset, ...]:
    if not items:
        raise ValueError("portfolio universe must be non-empty")
    assets: list[UniverseAsset] = []
    for raw in items:
        asset = UniverseAsset(
            symbol=str(raw["symbol"]),
            asset_class=str(raw["asset_class"]),
            min_weight=float(raw["min_weight"]),
            max_weight=float(raw["max_weight"]),
            price_required=bool(raw["price_required"]),
        )
        if asset.min_weight < 0:
            raise ValueError(f"{asset.symbol} min_weight must be non-negative")
        if asset.max_weight > 1:
            raise ValueError(f"{asset.symbol} max_weight must be <= 1")
        if asset.min_weight > asset.max_weight:
            raise ValueError(f"{asset.symbol} min_weight cannot exceed max_weight")
        assets.append(asset)
    return tuple(assets)


def _build_constraints(mapping: Mapping[str, Any]) -> PortfolioConstraints:
    constraints = PortfolioConstraints(
        gross_exposure_max=float(mapping["gross_exposure_max"]),
        net_exposure_min=float(mapping["net_exposure_min"]),
        net_exposure_max=float(mapping["net_exposure_max"]),
        single_asset_max=float(mapping["single_asset_max"]),
        turnover_threshold=float(mapping["turnover_threshold"]),
        rebalance_frequency=str(mapping["rebalance_frequency"]),
        leverage_allowed=bool(mapping["leverage_allowed"]),
        shorting_allowed=bool(mapping["shorting_allowed"]),
    )
    if not constraints.leverage_allowed and constraints.gross_exposure_max > 1.0 + EPSILON:
        raise ValueError("gross_exposure_max cannot exceed 1.0 when leverage is disabled")
    if constraints.shorting_allowed:
        raise ValueError("shorting_allowed=true requires an explicit shorting policy")
    return constraints


def _build_rho_buckets(items: list[Mapping[str, Any]]) -> tuple[RhoBucket, ...]:
    if not items:
        raise ValueError("rho_buckets must be non-empty")
    buckets = tuple(
        RhoBucket(
            name=str(item["name"]),
            lower=float(item["lower"]),
            upper=float(item["upper"]),
            lower_inclusive=bool(item["lower_inclusive"]),
            upper_inclusive=bool(item["upper_inclusive"]),
        )
        for item in items
    )
    ordered = sorted(buckets, key=lambda item: (item.lower, item.upper))
    if ordered[0].lower != 0.0 or not ordered[0].lower_inclusive:
        raise ValueError("rho buckets must start at 0.0 with inclusive lower bound")
    if ordered[-1].upper != 1.0 or not ordered[-1].upper_inclusive:
        raise ValueError("rho buckets must end at 1.0 with inclusive upper bound")
    for previous, current in zip(ordered, ordered[1:]):
        if previous.upper < current.lower - EPSILON:
            raise ValueError("rho buckets cannot have gaps")
        if previous.upper > current.lower + EPSILON:
            raise ValueError("rho buckets cannot overlap")
        if previous.upper_inclusive and current.lower_inclusive:
            raise ValueError("adjacent rho buckets cannot both include the shared boundary")
        if not previous.upper_inclusive and not current.lower_inclusive:
            raise ValueError("adjacent rho buckets cannot exclude the shared boundary")
    return tuple(ordered)


def _validate_state_policy(
    policy_mapping: Mapping[str, Any],
    universe: tuple[UniverseAsset, ...],
    constraints: PortfolioConstraints,
) -> Mapping[str, Mapping[str, Mapping[str, float]]]:
    universe_symbols = {asset.symbol for asset in universe}
    state_policy = policy_mapping["state_policy"]
    for state_name, bucket_mapping in state_policy.items():
        for bucket_name, weights in bucket_mapping.items():
            unknown = set(weights) - universe_symbols
            if unknown:
                raise ValueError(
                    f"state_policy bucket {state_name}/{bucket_name} references unknown assets {sorted(unknown)}"
                )
            weight_sum = float(sum(float(value) for value in weights.values()))
            if abs(weight_sum - 1.0) > EPSILON:
                raise ValueError(f"state_policy bucket {state_name}/{bucket_name} must sum to 1.0")
            for symbol, weight in weights.items():
                asset = next(item for item in universe if item.symbol == symbol)
                value = float(weight)
                if value < asset.min_weight - EPSILON or value > asset.max_weight + EPSILON:
                    raise ValueError(f"state_policy weight {symbol} violates asset min/max bounds")
                if value > constraints.single_asset_max + EPSILON:
                    raise ValueError(f"state_policy weight {symbol} exceeds single_asset_max")
    return state_policy


def load_portfolio_policy(path: str | Path) -> PortfolioPolicy:
    policy_path = Path(path)
    mapping = _read_policy_mapping(policy_path)
    _validate_bool_invariants(mapping)
    universe = _build_universe(list(mapping["universe"]))
    constraints = _build_constraints(mapping["constraints"])
    rho_buckets = _build_rho_buckets(list(mapping["rho_buckets"]))
    state_policy = _validate_state_policy(mapping, universe, constraints)
    return PortfolioPolicy(
        policy_id=str(mapping["policy_id"]),
        policy_version=int(mapping["policy_version"]),
        paper_only=True,
        broker_submission_allowed=False,
        base_currency=str(mapping["base_currency"]),
        universe=universe,
        constraints=constraints,
        rho_buckets=rho_buckets,
        state_policy=state_policy,
        execution_model=mapping["execution_model"],
        known_limitations=tuple(str(item) for item in mapping.get("known_limitations", [])),
    )
