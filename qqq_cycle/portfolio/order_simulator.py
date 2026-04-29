"""Phase 15 hypothetical order simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qqq_cycle.portfolio.delta import PortfolioDelta
from qqq_cycle.portfolio.policy import PortfolioPolicy
from qqq_cycle.portfolio.portfolio_snapshot import PaperPortfolioSnapshot
from qqq_cycle.portfolio.signal_gate import SignalEligibilityResult


@dataclass(frozen=True)
class HypotheticalOrder:
    order_id: str
    week_end: str
    symbol: str
    side: str
    quantity: float
    notional: float
    estimated_price: float
    slippage_bps: float
    estimated_slippage_cost: float
    commission: float
    estimated_total_cost: float
    reason: str
    paper_only: bool
    broker_submission_allowed: bool


@dataclass(frozen=True)
class OrderSimulationResult:
    week_end: str
    orders: tuple[HypotheticalOrder, ...]
    orders_count: int
    estimated_slippage_cost: float
    estimated_commission: float
    estimated_total_cost: float
    cash_before_orders: float
    cash_after_orders: float
    paper_only: bool
    broker_submission_allowed: bool
    reason: str
    friction_buffer_applied: bool


def simulate_hypothetical_orders(
    snapshot: PaperPortfolioSnapshot,
    delta: PortfolioDelta,
    signal_gate: SignalEligibilityResult,
    policy: PortfolioPolicy,
) -> OrderSimulationResult:
    if not signal_gate.execution_allowed or not delta.rebalance_required:
        return OrderSimulationResult(
            week_end=delta.week_end,
            orders=tuple(),
            orders_count=0,
            estimated_slippage_cost=0.0,
            estimated_commission=0.0,
            estimated_total_cost=0.0,
            cash_before_orders=float(snapshot.cash),
            cash_after_orders=float(snapshot.cash),
            paper_only=True,
            broker_submission_allowed=False,
            reason=signal_gate.reason if not signal_gate.execution_allowed else delta.reason,
            friction_buffer_applied=True,
        )

    execution_model = policy.execution_model
    slippage_bps = float(execution_model["slippage_bps_default"])
    commission = float(execution_model["commission_per_order"])
    min_notional = float(execution_model["min_notional"])
    friction_buffer_bps = float(execution_model["friction_buffer_bps"])
    total_friction_rate = slippage_bps / 10_000.0 + friction_buffer_bps / 10_000.0
    slippage_rate = slippage_bps / 10_000.0

    desired_notionals = {
        symbol: float(delta.delta_weights[symbol] * delta.nav)
        for symbol in delta.delta_weights
    }
    cash_available = float(snapshot.cash)
    orders: list[HypotheticalOrder] = []
    order_sequence = 1

    sell_symbols = [symbol for symbol, value in desired_notionals.items() if value < 0]
    for symbol in sorted(sell_symbols):
        desired_sell = abs(desired_notionals[symbol])
        if desired_sell < min_notional:
            continue
        price = float(snapshot.prices[symbol])
        quantity = desired_sell / price
        slippage_cost = desired_sell * slippage_rate
        total_cost = slippage_cost + commission
        cash_available += desired_sell - total_cost
        orders.append(
            HypotheticalOrder(
                order_id=f"paper-{delta.week_end}-{order_sequence:04d}",
                week_end=delta.week_end,
                symbol=symbol,
                side="SELL",
                quantity=quantity,
                notional=desired_sell,
                estimated_price=price,
                slippage_bps=slippage_bps,
                estimated_slippage_cost=slippage_cost,
                commission=commission,
                estimated_total_cost=total_cost,
                reason="rebalance_to_target",
                paper_only=True,
                broker_submission_allowed=False,
            )
        )
        order_sequence += 1

    buy_symbols = [symbol for symbol, value in desired_notionals.items() if value > 0]
    for symbol in sorted(buy_symbols):
        desired_buy = desired_notionals[symbol]
        max_buy_notional = max(0.0, (cash_available - commission) / (1.0 + total_friction_rate))
        actual_buy = min(desired_buy, max_buy_notional)
        if actual_buy < min_notional:
            continue
        price = float(snapshot.prices[symbol])
        quantity = actual_buy / price
        slippage_cost = actual_buy * slippage_rate
        total_cost = slippage_cost + commission
        cash_available -= actual_buy + total_cost
        cash_available = max(cash_available, 0.0)
        orders.append(
            HypotheticalOrder(
                order_id=f"paper-{delta.week_end}-{order_sequence:04d}",
                week_end=delta.week_end,
                symbol=symbol,
                side="BUY",
                quantity=quantity,
                notional=actual_buy,
                estimated_price=price,
                slippage_bps=slippage_bps,
                estimated_slippage_cost=slippage_cost,
                commission=commission,
                estimated_total_cost=total_cost,
                reason="rebalance_to_target",
                paper_only=True,
                broker_submission_allowed=False,
            )
        )
        order_sequence += 1

    return OrderSimulationResult(
        week_end=delta.week_end,
        orders=tuple(orders),
        orders_count=len(orders),
        estimated_slippage_cost=float(sum(order.estimated_slippage_cost for order in orders)),
        estimated_commission=float(sum(order.commission for order in orders)),
        estimated_total_cost=float(sum(order.estimated_total_cost for order in orders)),
        cash_before_orders=float(snapshot.cash),
        cash_after_orders=float(cash_available),
        paper_only=True,
        broker_submission_allowed=False,
        reason="orders_generated" if orders else "no_orders_generated",
        friction_buffer_applied=True,
    )
