# Phase 10 Acceptance

**Epoch**: `2021-03-30` → `2025-01-03`
**Strict rows**: 196  |  **High-confidence rows** (max_p_t ≥ 0.60): 58

---

## Part B — Regime Separation

**PASSED: True**

- `r1w`: KW p=0.0002, significant pairwise=4
- `R4w`: KW p=0.0000, significant pairwise=7
- `sigma4w`: KW p=0.0000, significant pairwise=6
- `mdd8w`: KW p=0.0000, significant pairwise=6

---

## Part A — Signal Predictive Power

**PASSED: True**

Key results (full subsample, fwd_vol and fwd_mdd at h=4w and h=8w):

| signal | horizon | target | spearman_rho | hac_pvalue | tercile_spread |
|--------|---------|--------|-------------|------------|----------------|
| `rho_t` | 4w | `fwd_vol_4w` | 0.582 | 0.000 | 0.0537 |
| `rho_t` | 4w | `fwd_mdd_4w` | 0.385 | 0.000 | 0.0411 |
| `rho_t` | 8w | `fwd_vol_8w` | 0.556 | 0.000 | 0.0659 |
| `rho_t` | 8w | `fwd_mdd_8w` | 0.359 | 0.001 | 0.0506 |
| `h_t` | 4w | `fwd_vol_4w` | 0.569 | 0.000 | 0.0551 |
| `h_t` | 4w | `fwd_mdd_4w` | 0.392 | 0.001 | 0.0450 |
| `h_t` | 8w | `fwd_vol_8w` | 0.539 | 0.003 | 0.0559 |
| `h_t` | 8w | `fwd_mdd_8w` | 0.380 | 0.011 | 0.0345 |
| `s_t` | 4w | `fwd_vol_4w` | 0.598 | 0.000 | 0.0526 |
| `s_t` | 4w | `fwd_mdd_4w` | 0.365 | 0.000 | 0.0384 |
| `s_t` | 8w | `fwd_vol_8w` | 0.564 | 0.000 | 0.0546 |
| `s_t` | 8w | `fwd_mdd_8w` | 0.339 | 0.000 | 0.0444 |

---

## Gate Decision

**Phase 11 permitted: True**

> Both B and A passed. Regime separation confirmed. Signal predictive power confirmed. Phase 11 may proceed.
