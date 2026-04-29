# Phase 15 Execution Sandbox Report

## Summary

- week_end: 2026-04-24
- phase14_snapshot_hash: a4303b47f434087de62b2bdb8417de75b2747ecc8f92a105cc999b4b1b7a5593
- signal_eligible: false
- execution_allowed: false
- target_generation_mode: hold_prior_or_policy_default
- rebalance_required: false
- orders_count: 0
- estimated_turnover: 0.0
- estimated_slippage_cost: 0.0
- estimated_commission: 0.0
- estimated_total_cost: 0.0
- paper_only: true
- broker_submission_allowed: false
- reason: degraded_backfill_signal

## Target Weights

{
  "BIL": 0.4,
  "QQQ": 0.6
}

## Delta

{
  "BIL": 0.0,
  "QQQ": 0.0
}

## Known Limitations

- 当前 Target Weights 采用阶梯式离散映射，在 rho_t 边界附近可能触发高换手。Phase 15 不优化该策略平滑问题，只记录其摩擦成本。

## Orders

- orders_count: 0
