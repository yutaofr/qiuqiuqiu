"""Phase 15 paper portfolio snapshot loading and validation.

Inputs:
    A CSV snapshot of current paper positions for one decision week.
Outputs:
    Immutable portfolio snapshot objects with validated NAV, weights, and cash.
Time semantics:
    The snapshot must represent holdings known at the stated weekly decision
    timestamp; this loader does not infer any later fills or prices.
As-of semantics:
    The loaded snapshot is treated as the current portfolio state as of
    `week_end` for sandbox-only order simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any, Iterable

import pandas as pd

from qqq_cycle.portfolio.policy import PortfolioPolicy


EPSILON = 1e-9


@dataclass(frozen=True)
class PositionSnapshot:
    account_id: str
    week_end: str
    symbol: str
    quantity: float
    market_price: float
    market_value: float
    weight: float
    source: str
    paper_only: bool
    broker_submission_allowed: bool


@dataclass(frozen=True)
class PaperPortfolioSnapshot:
    account_id: str
    week_end: str
    positions: tuple[PositionSnapshot, ...]
    weights: dict[str, float]
    quantities: dict[str, float]
    prices: dict[str, float]
    market_values: dict[str, float]
    cash: float
    nav: float
    source: str
    paper_only: bool
    broker_submission_allowed: bool


def _require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"portfolio snapshot missing required columns: {missing}")


def _finite_float(value: Any, field_name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


def load_portfolio_snapshot(
    path: str | Path,
    policy: PortfolioPolicy,
) -> PaperPortfolioSnapshot:
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"portfolio snapshot not found: {snapshot_path}")
    frame = pd.read_csv(snapshot_path)
    _require_columns(
        frame,
        {
            "account_id",
            "week_end",
            "symbol",
            "quantity",
            "market_price",
            "market_value",
            "weight",
            "cash",
            "source",
            "paper_only",
            "broker_submission_allowed",
        },
    )
    if frame.empty:
        raise ValueError("portfolio snapshot must contain at least one position row")

    account_ids = set(frame["account_id"].astype(str))
    if len(account_ids) != 1:
        raise ValueError("portfolio snapshot must contain exactly one account_id")
    week_ends = set(frame["week_end"].astype(str))
    if len(week_ends) != 1:
        raise ValueError("portfolio snapshot must contain exactly one week_end")
    sources = set(frame["source"].astype(str))
    if len(sources) != 1:
        raise ValueError("portfolio snapshot must contain exactly one source")

    cash_values = {_finite_float(value, "cash") for value in frame["cash"]}
    if len(cash_values) != 1:
        raise ValueError("portfolio snapshot cash field must be constant across rows")
    cash = next(iter(cash_values))
    if cash < -EPSILON:
        raise ValueError("portfolio snapshot cash cannot be negative")

    allowed_symbols = set(policy.symbols)
    positions: list[PositionSnapshot] = []
    weights: dict[str, float] = {}
    quantities: dict[str, float] = {}
    prices: dict[str, float] = {}
    market_values: dict[str, float] = {}

    for row in frame.to_dict(orient="records"):
        symbol = str(row["symbol"])
        if symbol not in allowed_symbols:
            raise ValueError(f"portfolio snapshot references unknown symbol {symbol}")
        paper_only = bool(row["paper_only"])
        broker_submission_allowed = bool(row["broker_submission_allowed"])
        if paper_only is not True:
            raise ValueError("portfolio snapshot must set paper_only=true")
        if broker_submission_allowed is not False:
            raise ValueError("portfolio snapshot must set broker_submission_allowed=false")

        quantity = _finite_float(row["quantity"], f"{symbol} quantity")
        price = _finite_float(row["market_price"], f"{symbol} market_price")
        market_value = _finite_float(row["market_value"], f"{symbol} market_value")
        weight = _finite_float(row["weight"], f"{symbol} weight")
        if price <= 0:
            raise ValueError(f"{symbol} market_price must be positive")
        if quantity < -EPSILON and not policy.constraints.shorting_allowed:
            raise ValueError(f"{symbol} negative quantity is not allowed")
        if market_value < -EPSILON:
            raise ValueError(f"{symbol} market_value cannot be negative")

        positions.append(
            PositionSnapshot(
                account_id=str(row["account_id"]),
                week_end=str(row["week_end"]),
                symbol=symbol,
                quantity=quantity,
                market_price=price,
                market_value=market_value,
                weight=weight,
                source=str(row["source"]),
                paper_only=True,
                broker_submission_allowed=False,
            )
        )
        weights[symbol] = weight
        quantities[symbol] = quantity
        prices[symbol] = price
        market_values[symbol] = market_value

    weight_sum = float(sum(weights.values()))
    if weight_sum < 0.99 or weight_sum > 1.01:
        raise ValueError("portfolio snapshot weights must sum to the interval [0.99, 1.01]")

    nav = float(sum(market_values.values()) + cash)
    if nav <= 0:
        raise ValueError("portfolio snapshot nav must be positive")

    return PaperPortfolioSnapshot(
        account_id=next(iter(account_ids)),
        week_end=next(iter(week_ends)),
        positions=tuple(positions),
        weights=weights,
        quantities=quantities,
        prices=prices,
        market_values=market_values,
        cash=cash,
        nav=nav,
        source=next(iter(sources)),
        paper_only=True,
        broker_submission_allowed=False,
    )
