# Phase 6 Integration Acceptance Report

## Outcome A — Strict Fixture Path

**Status: FAIL**


## Outcome B — Real Degraded Path

**Status: FAIL**


## Outcome C — Real Strict Path

**Status: PASS**

- Total rows: 1583
- Warmup rows: 733
- Strict rows: 196
- Degraded rows: 654
- h_t null in strict rows: 0 (expected 0)
- rho_t null in strict rows: 0 (expected 0)

## Acceptance Criteria Checklist

- [FAIL] strict_fixture_pipeline_output.csv exists
- [FAIL] degraded_real_pipeline_output.csv exists
- [PASS] strict_real_pipeline_output.csv exists
- [FAIL] fixture strict rows: h_t non-null
- [FAIL] fixture strict rows: rho_t non-null
- [FAIL] degraded post-warmup: h_t all null
- [FAIL] degraded rows: degraded_reason non-null
- [PASS] real strict rows: h_t non-null
- [PASS] real strict rows: rho_t non-null
- [PASS] pipeline_mode_summary.json written
