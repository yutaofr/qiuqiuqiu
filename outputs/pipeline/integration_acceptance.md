# Phase 6 Integration Acceptance Report

## Outcome A — Strict Fixture Path

**Status: PASS**

- Total rows: 1356
- Warmup rows: 523
- Strict rows: 833
- h_t null in strict rows: 0 (expected 0)
- rho_t null in strict rows: 0 (expected 0)

## Outcome B — Real Degraded Path

**Status: PASS**

- Total rows: 1583
- Warmup rows: 733
- Degraded rows: 850
- h_t non-null post-warmup: 0 (expected 0)
- rho_t non-null post-warmup: 0 (expected 0)
- k_hat_t non-null post-warmup: 848 (expected >0)

**Degraded reasons observed:**
- no contracts provided: h_t/rho_t unavailable

## Acceptance Criteria Checklist

- [PASS] strict_fixture_pipeline_output.csv exists
- [PASS] degraded_real_pipeline_output.csv exists
- [PASS] strict rows: h_t non-null
- [PASS] strict rows: rho_t non-null
- [PASS] degraded post-warmup: h_t all null
- [PASS] degraded rows: degraded_reason non-null
- [PASS] pipeline_mode_summary.json written
