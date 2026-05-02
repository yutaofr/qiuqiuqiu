# Phase 15 Execution Sandbox Report

## Summary

- week_end: 2026-05-01
- phase14_snapshot_hash: 
- signal_eligible: true
- execution_allowed: true
- target_generation_mode: policy_bucket_mapping
- rebalance_required: true
- orders_count: 2
- estimated_turnover: 0.4
- estimated_slippage_cost: 0.39960059910134793
- estimated_commission: 0.0
- estimated_total_cost: 0.39960059910134793
- paper_only: true
- broker_submission_allowed: false
- reason: orders_generated

## Target Weights

{
  "BIL": 0.8,
  "QQQ": 0.2
}

## Delta

{
  "BIL": 0.4,
  "QQQ": -0.39999999999999997
}

## Known Limitations

- 当前 Target Weights 采用阶梯式离散映射，在 rho_t 边界附近可能触发高换手。Phase 15 不优化该策略平滑问题，只记录其摩擦成本。

## Orders

- orders_count: 2
