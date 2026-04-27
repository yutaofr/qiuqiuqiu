# Implementation Plan: QQQ Cycle System — Remaining Second-Batch Modules

## Context

All first-batch modules (calendar, alignment, dual_memory, covariance, state_layer,
proto_online, stress_layer) and most second-batch infrastructure (pit_adjustment,
drift_probe, rolling_quantile, data_contracts) are already implemented and tested.

The remaining work covers exactly four areas, in dependency order:

1. **Config YAML** — hyperparameter single source of truth
2. **micro_layer.py** — fragility h_t and IIR lead with circuit breaker (§9)
3. **Production rho_t in risk_layer.py** — m_t, n_t, rho_t (§10.2–10.3)
4. **interpretability.py** — attribution, flags, diagnostics, health (§11)
5. **v1.1 test gaps** — two spec-named tests still missing

Nothing in Phase 3–4 can run until Phase 2 produces h_t^{lead}. Phase 5 is independent.

---

## Architecture Decisions

- `micro_layer.py` is a daily-frequency engine (τ-indexed) that aggregates to weekly (t-indexed). It consumes `PITAdjustmentEngine` for MA_20 and `WeightStore` + `ConstituentStore` for membership/weights. All of these data contracts already exist.
- `risk_layer.py` currently holds only `blended_state_weight()` (§10.1). Production rho_t (`compute_risk_score()`) will be added to the same file and gated by a `PRODUCTION_RISK_ENABLED` flag flip.
- `interpretability.py` is purely computational (no I/O) — it assembles the `InterpretabilityRecord` dataclass from already-computed layer outputs.
- `config/model_v22.yaml` is read-only at startup; no module should hard-code hyperparameters that are already in the spec.

---

## Task List

### Phase 1: Config Foundation

- [x] **Task 1: Create `qqq_cycle/config/model_v22.yaml`** — DONE 2026-04-27

  **Description:** Centralize all numeric hyperparameters from model-spec §4–10 into a single YAML. Add a `load_config()` loader in `config/__init__.py` (or a `config.py` shim) that returns a typed dataclass or namespace.

  **Acceptance criteria:**
  - [x] File exists at `qqq_cycle/config/model_v22.yaml` with keys for dual-memory windows (104, 260), covariance half-life (78), warmup (260), drift thresholds (1.2, 1.8), IIR delta (0.9), heal threshold (0.25), lambda_rho (0.75), omega_state ([1.0, 0.7, 0.3, 0.6, 0.9]), pct window (520), noise quantile (0.10)
  - [x] `load_config()` returns a typed object; callers can use dot access

  **Verification:**
  - [x] `python -c "from qqq_cycle.config import load_config; c = load_config(); assert c.warmup_weeks == 260"`

  **Dependencies:** None

  **Files:**
  - `qqq_cycle/config/model_v22.yaml` (new)
  - `qqq_cycle/config/__init__.py` (new)

  **Estimated scope:** XS

---

### Checkpoint: Phase 1

- [x] Config loads cleanly; no import errors

---

### Phase 2: Micro Layer

- [x] **Task 2: Core data structures and grace period (`micro_layer.py`)** — DONE 2026-04-27

  **Description:** Implement the daily-τ state object, grace period logic (§9.2), and matured member sets V^20 / V^60 (§9.3). These are pure data transformations — no MA_20 or correlation yet.

  **Acceptance criteria:**
  - [x] `MicroDailyState` dataclass tracks per-member age counters, grace period set, and grace expiry
  - [x] Grace period freezes breadth/correlation counters for missing members ≤ 3 days; sets c_t^{data}=1
  - [x] `V_tau_20` and `V_tau_60` correctly exclude sub-maturity members and members in grace period
  - [x] Giant missing weight check (§9.5): if `w_miss > 0.5 * w_(5)` → hold b_τ, c_τ; set c_t^{data}=1

  **Verification:**
  - [x] Unit test: member added on day 0 is not in V^20 until day 20
  - [x] Unit test: grace period member excluded from b_τ / c_τ for ≤3 missing days

  **Dependencies:** Task 1

  **Files:**
  - `qqq_cycle/core/micro_layer.py` (new)
  - `tests/test_micro_layer.py` (new)

  **Estimated scope:** M

- [x] **Task 3: Smoothed weights, breadth collapse, correlation concentration** — DONE 2026-04-27

  **Description:** Implement §9.6 smoothed lagged weights (ρ_w = 2^{-1/5}, frozen during rule windows), §9.7 weighted breadth b_τ (using PITAdjustmentEngine for MA_20^adj), and §9.8 weighted correlation concentration c_τ over V^60.

  **Acceptance criteria:**
  - [x] `compute_smoothed_weights()` uses ρ_w = 2^{-1/5}; freezes updates on rule/rebalance days
  - [x] `compute_breadth()` calls `pit_engine.get_adjusted_window(ticker, τ, 20, asof=τ)` for each member in V^20; if PITAdjustmentEngine unavailable → micro layer halts (raises `MicroLayerUnavailableError`)
  - [x] `compute_correlation_concentration()` computes normalized weighted average off-diagonal correlation per §9.8 formula
  - [x] Weekly aggregation: `b_t^{wk}` = median of τ in week(t), same for `c_t^{wk}` (§9.9)

  **Verification:**
  - [x] Unit test: 100% of stocks above MA_20 → b_τ = 0 (no fragility)
  - [x] Unit test: all stocks perfectly correlated → c_τ = 1
  - [x] Unit test: `get_adjusted_window` unavailable → `MicroLayerUnavailableError` raised

  **Dependencies:** Task 2

  **Files:**
  - `qqq_cycle/core/micro_layer.py` (extend)
  - `tests/test_micro_layer.py` (extend)

  **Estimated scope:** M

- [x] **Task 4: Rule-week downweighting, micro score, and IIR circuit breaker** — DONE 2026-04-27

  **Description:** Implement §9.10 rule-week weighted robust z-score (`z^{wrob}`), §9.11 micro score M_t^{raw}, logistic mapping h_t, and the IIR envelope with 3-consecutive-week circuit breaker (h_t^{lead}).

  **Acceptance criteria:**
  - [x] `z_wrob_156()` applies ω_τ = 0.3 for rule/rebalance weeks; ω_τ = 1.0 otherwise
  - [x] `iir_envelope_with_breaker()` matches spec §6.3 signature exactly: if h_t < 0.25 for 3 consecutive weeks → x_t^{lead} = 0
  - [x] `h_t^{lead} = 0.5 + x_t^{lead}` ∈ [0.5, 1.0] when no fragility signal; ∈ [0.5, 1.0+] only while IIR decays
  - [x] Single-week sub-threshold does NOT clear IIR memory

  **Verification:**
  - [x] Unit test: 2-week recovery (h_t = 0.1, 0.1) does not clear IIR → h_t^{lead} remains elevated
  - [x] Unit test: 3-week recovery (h_t = 0.1, 0.1, 0.1) clears IIR → h_t^{lead} resets to 0.5
  - [x] Unit test: δ=0.9 decay correct over 10 weeks with no new signal

  **Dependencies:** Task 3

  **Files:**
  - `qqq_cycle/core/micro_layer.py` (extend)
  - `tests/test_micro_layer.py` (extend, targeting test_micro_point_in_time.py)

  **Estimated scope:** S

---

### Checkpoint: Phase 2

- [x] `pytest tests/test_micro_layer.py` passes
- [x] `pytest tests/test_micro_point_in_time.py` passes (PIT semantics)
- [x] No imports of `micro_layer` fail

---

### Phase 3: Production Risk Score

- [x] **Task 5: Full rho_t in `risk_layer.py`** — DONE 2026-04-27

  **Description:** Extend `risk_layer.py` with `compute_risk_score()` implementing §10.2 (m_t, n_t) and §10.3 (rho_t formula). Flip `PRODUCTION_RISK_ENABLED = True`. Add EWCorr_78w tracker for η_t (§10.4) that feeds interpretability only.

  **Acceptance criteria:**
  - [x] `compute_risk_score(omega_t, s_t, h_t_lead, lambda_rho=0.75)` returns rho_t ∈ [0, 1]
  - [x] rho_t monotonically increasing in s_t and h_t_lead
  - [x] rho_t = 0 when s_t = 0 and h_t_lead = 0.5 (no micro signal)
  - [x] `PRODUCTION_RISK_ENABLED = True`
  - [x] `compute_ewcorr_78w(s_series, h_lead_series)` returns η_t series (for interpretability only)

  **Verification:**
  - [x] Unit test: rho_t = 1 - (1-1.0)(1-0.75*1.0) = 1.0 at maximum inputs
  - [x] Unit test: rho_t ∈ [0,1] over 1000 random (m_t, n_t) pairs

  **Dependencies:** Task 4

  **Files:**
  - `qqq_cycle/core/risk_layer.py` (extend)
  - `tests/test_risk_layer.py` (extend)

  **Estimated scope:** S

---

### Checkpoint: Phase 3

- [x] `pytest tests/test_risk_layer.py` passes
- [x] `PRODUCTION_RISK_ENABLED is True`

---

### Phase 4: Interpretability Module

- [x] **Task 6: `interpretability.py`** — DONE 2026-04-27

  **Description:** Implement `InterpretabilityRecord` dataclass and `build_interpretability()` assembler per §11. All attribution, contamination, drift diagnostic, and health fields.

  **Acceptance criteria:**
  - [x] `InterpretabilityRecord` has fields: `A_t` (attribution), `C_t` (contamination flags), `D_t` (drift diagnostics), `H_t` (module health)
  - [x] `A_t.H_components = (0.40*L_t, 0.35*T_t, 0.25*P_t)` etc. per §11.1
  - [x] Contamination flags: c_t^{rule}, c_t^{const}, c_t^{data}, c_t^{micro}, c_t^{drift} per §11.2
  - [x] Module health: h_t^{macro}, h_t^{exo}, h_t^{micro}, h_t^{state} per §11.4
  - [x] Pure computation — no I/O in this module

  **Verification:**
  - [x] Unit test: `build_interpretability()` round-trips all fields without loss
  - [x] Unit test: `c_t^{drift} = 1` when `|delta_abs_raw| >= 1.8`

  **Dependencies:** Task 5

  **Files:**
  - `qqq_cycle/core/interpretability.py` (new)
  - `tests/test_interpretability.py` (new)

  **Estimated scope:** M

---

### Checkpoint: Phase 4

- [x] `pytest tests/test_interpretability.py` passes
- [ ] Full system output tuple `(k_hat_t, p_t, s_t, h_t, rho_t, I_t)` producible end-to-end in strict mode
  - NOTE 2026-04-27: Module-level producers for `h_t`, `rho_t`, and `I_t` are implemented and tested, but no strict end-to-end pipeline wiring exists in this slice.

---

### Phase 5: v1.1 Test Coverage Gaps

- [x] **Task 7: Add two missing spec-named tests** — DONE 2026-04-27

  **Description:** The spec (§4.4, §5.7) names specific test functions that are still absent or named differently.

  **Acceptance criteria:**
  - [x] `tests/test_covariance.py` contains `test_condition_threshold_reachable`: drives `condition_number_reg` above `COND_WARN_RATIO / eps_rel` (= 9000) with a near-degenerate input
  - [x] `tests/test_dual_memory.py` contains `test_nan_does_not_propagate_to_covariance`: feeds a NaN-output z_ew series into `RobustEWCov2D.update()` and asserts state is unchanged

  **Verification:**
  - [x] `pytest tests/test_covariance.py::test_condition_threshold_reachable` passes
  - [x] `pytest tests/test_dual_memory.py::test_nan_does_not_propagate_to_covariance` passes

  **Dependencies:** None (independent of Phases 2–4)

  **Files:**
  - `tests/test_covariance.py` (add function)
  - `tests/test_dual_memory.py` (add function)

  **Estimated scope:** XS

---

### Checkpoint: Phase 5 (Final)

- [x] `pytest` (full suite) passes with no failures
- [x] All spec-named tests from §4.4 and §5.7 present and passing

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `PITAdjustmentEngine` not available in test environment | High | Gate Task 3 tests with a mock engine stub that returns synthetic 20-point windows |
| Weighted robust z-score (`z^{wrob}`) with rule-week ω=0.3 changes distribution shape | Med | Compare against uniform-weight z-score; assert it falls within ±20% of scale at long horizons |
| IIR circuit breaker off-by-one (2 weeks vs 3) | Med | Parameterize N_heal=3 in test; assert boundary at exactly 3 consecutive weeks |
| micro_layer τ→t aggregation creates forward-looking leakage | High | Test: weekly median uses only days within calendar week; no future-week days included |

## Open Questions

None — all spec decisions are locked in v1.1/v2.2-final. The two NL items (NL-1, NL-2) are monitoring items, not implementation blockers.
