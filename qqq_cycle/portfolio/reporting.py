"""Phase 15 sandbox artifact writing and report rendering."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from qqq_cycle.portfolio.delta import PortfolioDelta
from qqq_cycle.portfolio.order_simulator import OrderSimulationResult
from qqq_cycle.portfolio.target_weights import TargetWeightsResult


KNOWN_LIMITATION_TEXT = (
    "当前 Target Weights 采用阶梯式离散映射，在 rho_t 边界附近可能触发高换手。"
    "Phase 15 不优化该策略平滑问题，只记录其摩擦成本。"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=True)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_jsonable(raw) for key, raw in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(raw) for key, raw in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(_to_jsonable(payload)) + "\n", encoding="utf-8")


def _write_orders_csv(path: Path, orders: OrderSimulationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "order_id,week_end,symbol,side,quantity,notional,estimated_price,slippage_bps,estimated_slippage_cost,commission,estimated_total_cost,reason,paper_only,broker_submission_allowed"
    ]
    for order in orders.orders:
        lines.append(
            ",".join(
                [
                    order.order_id,
                    order.week_end,
                    order.symbol,
                    order.side,
                    f"{order.quantity:.10f}",
                    f"{order.notional:.10f}",
                    f"{order.estimated_price:.10f}",
                    f"{order.slippage_bps:.4f}",
                    f"{order.estimated_slippage_cost:.10f}",
                    f"{order.commission:.10f}",
                    f"{order.estimated_total_cost:.10f}",
                    order.reason,
                    str(order.paper_only).lower(),
                    str(order.broker_submission_allowed).lower(),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_execution_summary(
    *,
    week_end: str,
    phase14_snapshot: Mapping[str, Any],
    target: TargetWeightsResult,
    delta: PortfolioDelta,
    orders: OrderSimulationResult,
) -> dict[str, Any]:
    known_limitations = list(dict.fromkeys(target.known_limitations))
    return {
        "week_end": week_end,
        "phase14_snapshot_hash": str(phase14_snapshot.get("source_hash") or ""),
        "signal_eligible": target.signal_eligible,
        "execution_allowed": delta.reason not in {"degraded_backfill_signal", "block_signal", "not_strict_mode", "execution_not_permitted", "strict_gate_failed", "h_t_missing", "rho_t_missing", "k_hat_t_missing", "s_t_missing", "paper_only_invariant_failed"},
        "target_generation_mode": target.generation_mode,
        "rebalance_required": delta.rebalance_required,
        "orders_count": orders.orders_count,
        "estimated_turnover": delta.turnover,
        "estimated_slippage_cost": orders.estimated_slippage_cost,
        "estimated_commission": orders.estimated_commission,
        "estimated_total_cost": orders.estimated_total_cost,
        "paper_only": True,
        "broker_submission_allowed": False,
        "reason": delta.reason if not delta.rebalance_required else orders.reason,
        "known_limitations": known_limitations,
    }


def render_execution_sandbox_report(
    summary: Mapping[str, Any],
    target: TargetWeightsResult,
    delta: PortfolioDelta,
    orders: OrderSimulationResult,
) -> str:
    return "\n".join(
        [
            "# Phase 15 Execution Sandbox Report",
            "",
            "## Summary",
            "",
            f"- week_end: {summary['week_end']}",
            f"- phase14_snapshot_hash: {summary['phase14_snapshot_hash']}",
            f"- signal_eligible: {str(summary['signal_eligible']).lower()}",
            f"- execution_allowed: {str(summary['execution_allowed']).lower()}",
            f"- target_generation_mode: {summary['target_generation_mode']}",
            f"- rebalance_required: {str(summary['rebalance_required']).lower()}",
            f"- orders_count: {summary['orders_count']}",
            f"- estimated_turnover: {summary['estimated_turnover']}",
            f"- estimated_slippage_cost: {summary['estimated_slippage_cost']}",
            f"- estimated_commission: {summary['estimated_commission']}",
            f"- estimated_total_cost: {summary['estimated_total_cost']}",
            f"- paper_only: {str(summary['paper_only']).lower()}",
            f"- broker_submission_allowed: {str(summary['broker_submission_allowed']).lower()}",
            f"- reason: {summary['reason']}",
            "",
            "## Target Weights",
            "",
            json.dumps(target.target_weights, sort_keys=True, ensure_ascii=True, indent=2),
            "",
            "## Delta",
            "",
            json.dumps(delta.delta_weights, sort_keys=True, ensure_ascii=True, indent=2),
            "",
            "## Known Limitations",
            "",
            f"- {KNOWN_LIMITATION_TEXT}",
            "",
            "## Orders",
            "",
            f"- orders_count: {orders.orders_count}",
        ]
    ) + "\n"


def write_phase15_artifacts(
    *,
    output_dir: str | Path,
    week_end: str,
    phase14_snapshot: Mapping[str, Any],
    target: TargetWeightsResult,
    delta: PortfolioDelta,
    orders: OrderSimulationResult,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target_path = out / f"target_weights_{week_end}.json"
    delta_path = out / f"portfolio_delta_{week_end}.json"
    orders_path = out / f"hypothetical_orders_{week_end}.csv"
    report_path = out / f"execution_sandbox_report_{week_end}.md"
    summary_latest_path = out / "execution_sandbox_summary_latest.json"

    summary = build_execution_summary(
        week_end=week_end,
        phase14_snapshot=phase14_snapshot,
        target=target,
        delta=delta,
        orders=orders,
    )
    summary["report_sha256"] = hashlib.sha256(
        render_execution_sandbox_report(summary, target, delta, orders).encode("utf-8")
    ).hexdigest()

    _write_json(target_path, target)
    _write_json(delta_path, delta)
    _write_orders_csv(orders_path, orders)
    report_path.write_text(
        render_execution_sandbox_report(summary, target, delta, orders),
        encoding="utf-8",
    )
    _write_json(summary_latest_path, summary)
    return {
        "target_weights": target_path,
        "portfolio_delta": delta_path,
        "hypothetical_orders": orders_path,
        "execution_sandbox_report": report_path,
        "execution_sandbox_summary_latest": summary_latest_path,
    }
