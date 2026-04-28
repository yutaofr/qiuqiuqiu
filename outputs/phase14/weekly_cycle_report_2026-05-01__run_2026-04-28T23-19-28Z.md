# Weekly Cycle Report

## Snapshot

- week_end: 2026-05-01
- published_at: 2026-04-28T23:19:28Z
- operational_sla_cutoff: SAT 12:00 America/New_York
- source_hash: 7291a9bc86c8ce51ec01d4a0a7b7fe47e70f2191acc5f7ca58a6091a3ac09f06
- mode: degraded
- k_hat_t: null
- p_t: null
- s_t: null
- h_t: null
- rho_t: null
- drift_flag: 0
- execution_state: block
- execution_permitted: false
- signal_valid_but_not_executable: false
- strict_contracts_satisfied: false
- degraded_reason: h_t unavailable for this week: micro data window not satisfied
- execution_block_reason: constituent snapshot unavailable for 2026-05-01: no constituent data for trade_date=2026-05-01 asof=2026-05-01 00:00:00; weight snapshot unavailable for 2026-05-01: no weight data for trade_date=2026-05-01 asof=2026-05-01 00:00:00

## Freshness

- fred_macro: fresh_enough=True, blocking_level=degrade, last_observation_date=2026-05-01, reason=None
- ai_gpr: fresh_enough=False, blocking_level=degrade, last_observation_date=2026-04-03, reason=AI_GPR last obs 2026-04-03 < week_end 2026-05-01
- qqq_prices: fresh_enough=True, blocking_level=block, last_observation_date=2026-05-01, reason=None
- constituents: fresh_enough=False, blocking_level=block, last_observation_date=unknown, reason=constituent snapshot unavailable for 2026-05-01: no constituent data for trade_date=2026-05-01 asof=2026-05-01 00:00:00
- weights: fresh_enough=False, blocking_level=block, last_observation_date=unknown, reason=weight snapshot unavailable for 2026-05-01: no weight data for trade_date=2026-05-01 asof=2026-05-01 00:00:00
- pit_prices: fresh_enough=True, blocking_level=block, last_observation_date=2026-05-01, reason=None
