# First Auditable QQQ State Slice Notes

## Compliance

- Package code lives under `qqq_cycle/core/` to avoid a root-level `core` namespace collision.
- `.env` loading is implemented by `load_fred_api_key`; the key is checked but never printed.
- Alignment helpers use backward as-of semantics only: observations after a decision timestamp are ineligible.
- Dual-memory normalization implements the specified robust z-score, EW z-score, exogenous pre-transform, Huber variance EW z-score, clipping, and NaN warmup policy.
- Covariance uses static sample covariance plus `eps_abs * I` for cold start and selective spectral flooring for all 2D covariance matrices.
- Mahalanobis distance uses previous `cov_reg`, matching the engineering override in `docs/qqq_archi_spec_v1.1.md`.

## Conservative Choices

- `Lambda` is defined as the standard logistic function because the model spec names it but does not define a different map.
- NaN rows are not imputed. For EW recursions with elapsed missing weeks, the next valid update uses `effective_rho = rho ** k` and the complement weight `1 - effective_rho`, preserving total EW weight mass.
- Pandas 3.0.1 rolling behavior is pinned in `pyproject.toml`; tests cover `min_periods` and NaN masking behavior used by dual-memory normalization.
- Replay tables are diagnostic only. They include state/stress probabilities and labels, but they do not compute returns, micro-layer `h_t`, risk-layer `rho_t`, or production risk decisions.
- The drift probe is the minimum physical-space implementation from the v2.2 spec: 520-week rolling empirical percentiles, EW physical `H_raw`, rolling median/MAD baseline, and `|drift_probe_raw| >= 1.8` flagging.
- `InMemoryPITAdjustmentEngine` is a deterministic fixture engine only. It validates as-of filtering and relative-basis scaling, but it is not a production corporate-action backfill engine.
- `blended_state_weight` is implemented as an interface utility from model §10.1. No production `rho_t` output is implemented.
- Phase 14 regime monitoring uses the latest immutable snapshot per `week_end` and keys monitored regimes conservatively as `mode` plus `k_hat_t` when available (for example `strict:k2`). This avoids conflating degraded and strict observations that share the same state index.

## Performance And MLX

- MLX is not used in this slice. The workload is pandas time alignment and small 2D NumPy linear algebra, where MLX would add complexity without material acceleration.
- The exact 520-week rolling/expanding quantile in the stress noise floor is acceptable for this smoke slice. Production-scale histories should evaluate a streaming quantile structure such as T-Digest or scheduled lower-frequency recomputation.

## Remaining Blockers

- Real FRED fetching is not implemented in this slice.
- The PIT adjusted-close engine remains a contract plus fixture implementation; no real corporate-action backfill source is implemented.
- The micro layer remains blocked until the PIT adjustment engine and historical constituent/weight data contracts are implemented.
- Risk layer production outputs and full historical return backtests are out of scope.

## Sample Diagnostic Log Output

```json
{
  "maha": 1.24,
  "huber_weight": 1.0,
  "eigval_1": 0.83,
  "eigval_2_raw": 0.00002,
  "eigval_2_reg": 0.000083,
  "condition_number_raw": 41500.0,
  "condition_number_reg": 10000.0,
  "eigval_2_was_floored": true
}
```
