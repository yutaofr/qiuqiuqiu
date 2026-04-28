# Ops Status Summary

Current status: BLOCK
Reason: h_t unavailable for this week: micro data window not satisfied
Operator action: see docs/OPS_RUNBOOK.md §2.1

## Snapshot

- required_week_end: 2026-04-24
- latest_available_week_end: 2026-05-01
- published_at: 2026-04-28T23:19:28Z
- current_mode: degraded
- operational_sla_cutoff: SAT 12:00 America/New_York

## Operational Dimensions

- signal_validity: OK (none)
- execution_readiness: OK (none)
- data_health: BLOCK (required week_end 2026-04-24 is not published; latest available week_end is 2026-05-01)

## Runbook

- path: docs/OPS_RUNBOOK.md
- references: §2.1

## Alerts

- [BLOCK] data_health / missing_required_snapshot: required week_end 2026-04-24 is not published; latest available week_end is 2026-05-01 (§2.1)
