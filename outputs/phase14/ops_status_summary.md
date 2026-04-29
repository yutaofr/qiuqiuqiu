# Ops Status Summary

Current status: BLOCK
Reason: controlled block: micro IIR state held
Operator action: see docs/OPS_RUNBOOK.md §2.2

## Snapshot

- required_week_end: 2026-04-24
- latest_available_week_end: 2026-04-24
- published_at: 2026-04-29T09:50:06Z
- current_mode: degraded
- operational_sla_cutoff: SAT 12:00 America/New_York

## Operational Dimensions

- signal_validity: BLOCK (signal tuple is incomplete for the required week_end)
- execution_readiness: BLOCK (constituent_store not provided; weight_store not provided; pit_engine not provided)
- data_health: BLOCK (constituents stale: constituent_store not provided)

## Runbook

- path: docs/OPS_RUNBOOK.md
- references: §2.2, §2.3, §3.2, §3.3

## Alerts

- [BLOCK] data_health / stale_source_block: constituents stale: constituent_store not provided (§2.3)
- [BLOCK] data_health / stale_source_block: weights stale: weight_store not provided (§2.3)
- [BLOCK] data_health / stale_source_block: pit_prices stale: pit_engine not provided (§2.3)
- [BLOCK] execution_readiness / execution_blocked: constituent_store not provided; weight_store not provided; pit_engine not provided (§3.2)
- [BLOCK] signal_validity / signal_invalid: signal tuple is incomplete for the required week_end (§3.3)
- [DEGRADE] data_health / stale_source_degrade: ai_gpr stale: AI_GPR last obs 2026-04-03 < week_end 2026-04-24 (§2.3)
- [WARN] data_health / late_snapshot_publication: snapshot for week_end 2026-04-24 published at 2026-04-29T09:50:06Z after SLA cutoff 2026-04-25T16:00:00Z (§2.2)
