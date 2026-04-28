# Strict Real Path Coverage Note

## What this path covers

- **Data scope**: partial real seeded micro data only
- **Micro source range**: 2021-01-01 to 2024-12-31 (1004 trading days)
- **Seeded ticker count**: 20 tickers
- **Strict rows produced**: 196
- **Strict week range**: 2021-04-09 to 2025-01-03

## What this path does NOT cover

- This is NOT full historical QQQ micro coverage.
- This is NOT a production-grade strict path.
- The seeded universe is a partial subset of QQQ constituents.
- Prices are seeded from a fixed date range; no live feed is wired.

## Purpose

This path validates engineering wiring only:
- PIT constituent + weight + price stores connect correctly to the pipeline
- Daily micro loop (breadth, correlation) produces h_t for the seeded period
- Strict rows appear where micro data coverage is satisfied
- Pipeline cuts to degraded when micro data ends

This path does NOT authorize strategy deployment or production release.

## Evidence grade

- strict_data_scope: partial_real_seeded
- strict_real_contract_grade: conditional
- strict_real_production_eligible: false
- phase_7_verdict: pass_conditional

These fields are verifiable in pipeline_mode_summary.json.
