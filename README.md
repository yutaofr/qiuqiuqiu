# QQQ Cycle Intelligence Operations

This repository implements QQQ Cycle Intelligence Operations:

- weekly cycle state monitoring
- PIT controlled backfill
- publication proof evaluation
- canonical namespace normalization
- Phase 14 snapshot publishing / revision audit / ops status
- Phase 15 paper-only portfolio and execution sandbox

This repository is not a live trading system.
It produces auditable research, operations, and paper-only sandbox artifacts.

## Current Status

Current accepted state:

| Field | Value |
|---|---|
| `selected_scheme` | `degraded_backfill` |
| `proof_strict_eligible` | `false` |
| `strict_eligibility_reason` | `evidence_after_sla_cutoff` |
| `strict_validation_ok` | `true` |
| `degraded_validation_ok` | `true` |
| `Phase 15 sandbox status` | `no-trade degraded state` |
| `orders_count` | `0` |
| `paper_only` | `true` |
| `broker_submission_allowed` | `false` |

The system is not in `strict_recovery` for `2026-04-24` because no machine-verifiable before-SLA publication proof exists.

`degraded_backfill` is the correct state when data validation passes but strict PIT proof fails.

Phase 15 intentionally generates zero orders for degraded signals.

Current machine-readable evidence lives in:

- `outputs/phase14/ops_status_summary.json`
- `outputs/phase14/weekly_cycle_report_latest.md`
- `outputs/phase14/cycle_snapshot_latest.json`
- `outputs/phase15/execution_sandbox_summary_latest.json`
- `outputs/phase15/execution_sandbox_report_2026-04-24.md`

## Project Overview

This repository is the operational and research home for QQQ cycle-state monitoring.
It is built to support point-in-time data handling, controlled backfills, proof-based publication gating, and paper-only execution diagnostics.

The intended outputs are human-auditable Markdown, JSON, and CSV artifacts.
The system does not submit live orders.

## Operator Weekly Workflow

For an Operator / PM, the intended weekly workflow is:

| Step | What to check | What it means |
|---|---|---|
| 1 | Ops Status | `strict_recovery` means PIT proof verified and strict validation passed. `degraded_backfill` means validation passed but PIT proof is insufficient. `block` means source, normalization, validation, or control contract failed. |
| 2 | Signal Diagnostics | Review `k_hat_t`, `rho_t`, and `s_t` in the weekly report. These are diagnostics, not trade instructions. |
| 3 | Phase 15 Sandbox | Review target weights, portfolio delta, hypothetical paper orders, and estimated costs. |
| 4 | Human decision | Use the report as decision support only. The system never submits orders. |

Do not hand-edit JSON or CSV outputs.
Do not connect the repository directly to a broker API.

Cron, email digest, and dashboard are operational roadmap items unless corresponding scripts or configs exist.
Current supported path is command-line execution plus generated Markdown/JSON/CSV artifacts.

## Status Model: Strict / Degraded / Block

| Status | Meaning | Store Path | Phase 15 Behavior |
|---|---|---|---|
| `strict_recovery` | PIT proof verified and strict validation passed | `stores/strict` | May generate paper-only hypothetical orders if the signal gate passes |
| `degraded_backfill` | Validation passed but PIT proof failed | `stores/backfill` | Hold prior target; `orders_count=0` |
| `block` | Source, normalization, validation, or control failed | no store write | No signal; `orders_count=0` |

`strict_recovery` requires `evaluate_publication_proof(...)` to compute `strict_eligible=true`.
A file existing on disk is not enough to pass strict.

## Signal Interpretation

The core signal fields are:

- `k_hat_t`: inferred macro cycle state
- `rho_t`: micro fragility / break risk score
- `s_t`: macro stress score
- `h_t`: micro observation / input layer signal

If `rho_t` is `null`, `h_t` is `null`, or `micro_state_frozen=true`, Phase 15 must not generate execution orders.

For `2026-04-24`:

- `h_t = null`
- `rho_t = null`
- `micro_state_frozen = true`
- therefore Phase 15 generates zero orders

These fields are diagnostics and gate inputs. They are not a direct instruction to buy or sell.

## Phase 15 Execution Sandbox

Phase 15 is paper-only.

Hard invariants:

- `paper_only = true`
- `broker_submission_allowed = false`
- no broker API
- no account authentication
- no live order submission
- synthetic strict tests must use `tmp_path` or in-memory mocks only

Expected artifacts:

- `outputs/phase15/target_weights_2026-04-24.json`
- `outputs/phase15/portfolio_delta_2026-04-24.json`
- `outputs/phase15/hypothetical_orders_2026-04-24.csv`
- `outputs/phase15/execution_sandbox_report_2026-04-24.md`
- `outputs/phase15/execution_sandbox_summary_latest.json`

Current degraded expectation:

```json
{
  "paper_only": true,
  "broker_submission_allowed": false,
  "signal_eligible": false,
  "execution_allowed": false,
  "orders_count": 0,
  "reason": "degraded_backfill_signal"
}
```

Phase 15 artifacts are for audit and decision support only.
They are not broker instructions.

## Safety Invariants

- Treat all time semantics as point-in-time.
- Never use hindsight-adjusted data where a PIT contract is required.
- Weekly decisions may only use data knowable by the decision timestamp.
- If point-in-time adjusted prices are unavailable, the micro layer must stop or degrade gracefully.
- Never silently fallback on numerical linear algebra failures.
- Log numerical health metrics and degrade module health on failure.
- Do not change model math, state labels, windows, half-lives, thresholds, or weights unless a task explicitly asks for it.
- Do not fabricate strict evidence, publication proofs, or broker readiness.
- Do not manually edit canonical JSON/CSV artifacts to simulate a passing state.

## Repository Layout

| Path | Purpose |
|---|---|
| `qqq_cycle/` | Python package with core, data contract, ops, live, backtest, and portfolio modules |
| `scripts/` | CLI entrypoints for capture, backfill, ops, publishing, replay, and sandbox runs |
| `tests/` | Unit, integration, and regression tests for point-in-time, proof, ops, and sandbox behavior |
| `docs/` | Model spec, architecture spec, runbook, assumptions, and contract notes |
| `outputs/` | Generated Markdown, JSON, CSV, and replay artifacts |
| `stores/` | Persistent backfill and strict-store artifacts |
| `data/` | Input data and supporting raw or normalized assets |
| `configs/` | Repository configuration files |
| `state/` | Operational state snapshots and runtime state helpers |
| `normalized/` | Normalized data products |
| `sandbox/` | Local sandbox scratch outputs |
| `captures/` | Captured source files or extracts |
| `conductor/` | Orchestration and workflow support |

The key implementation references are:

- `docs/model-spec.md`
- `docs/qqq_archi_spec_v1.1.md`
- `docs/OPS_RUNBOOK.md`
- `docs/PIT_ADJUSTED_CLOSE_CONTRACT.md`
- `docs/ASSUMPTIONS.md`

## Environment Setup

Repository requirements:

- Python `>=3.13`
- `numpy==1.26.4`
- `pandas==3.0.1`
- `pytest==9.0.2`
- `python-dotenv==1.1.0`

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The FRED API key is expected in `.env`.
Do not hardcode credentials.

## Quick Start: Verify Current State

1. Open `outputs/phase14/ops_status_summary.json`.
2. Confirm `selected_scheme = degraded_backfill`.
3. Confirm `proof_strict_eligible = false`.
4. Confirm `current_status = BLOCK`.
5. Open `outputs/phase15/execution_sandbox_summary_latest.json`.
6. Confirm `paper_only = true`, `broker_submission_allowed = false`, and `orders_count = 0`.
7. Open `outputs/phase15/execution_sandbox_report_2026-04-24.md`.
8. Confirm the report says `reason: degraded_backfill_signal`.

If those checks do not match, stop and inspect the upstream data and proof contract before doing anything else.

## Full Replay Workflow

Use the replay and audit scripts when you need to reproduce historical behavior or validate invariant behavior against archived data.

Common entrypoints:

- `scripts/run_replay_tables.py`
- `scripts/run_real_replay.py`
- `scripts/run_state_stress_audit.py`
- `scripts/audit_phase_x_final_replay.py`

Expected outputs are under:

- `outputs/replay/real/`
- `outputs/replay/synthetic/`
- `outputs/audit/state_stress_replay/`

Replay outputs are for validation and audit only.
They are not execution instructions.

## Strict Evidence Upgrade Workflow

Use this workflow when attempting to move from degraded evidence toward strict eligibility.

Typical steps:

1. Capture strict evidence with `scripts/capture_strict_evidence.py`.
2. Run controlled backfill with `scripts/run_controlled_backfill.py`.
3. Rebuild revision evidence with `scripts/run_phase14_revision_audit.py`.
4. Publish the Phase 14 snapshot with `scripts/run_phase14_publish.py`.
5. Evaluate publication proof through `evaluate_publication_proof(...)`.

Strict eligibility only exists when the proof is machine-verifiable and before the SLA cutoff.
If the publication proof lands after the cutoff, the correct outcome remains `degraded_backfill`.

Do not treat a file on disk as proof.

## Phase 15 Sandbox Workflow

Use the Phase 15 sandbox to inspect paper-only portfolio and execution diagnostics.

Typical steps:

1. Run `scripts/run_phase15_sandbox.py`.
2. Inspect `outputs/phase15/execution_sandbox_summary_latest.json`.
3. Inspect `outputs/phase15/execution_sandbox_report_2026-04-24.md`.
4. Review target weights, portfolio delta, and hypothetical order count.

Phase 15 does not authenticate to a broker.
Phase 15 does not submit orders.
Phase 15 does not create live account activity.

## Expected Outputs

Phase 14:

- `outputs/phase14/ops_status_summary.json`
- `outputs/phase14/ops_status_summary.md`
- `outputs/phase14/weekly_cycle_report_latest.md`
- `outputs/phase14/cycle_snapshot_latest.json`
- `outputs/phase14/state_transition_matrix.csv`
- `outputs/phase14/revision_stability_summary.csv`
- `outputs/phase14/revision_stability_tests.json`
- `outputs/phase14/strict_evidence_inventory_2026-04-24.json`

Phase 15:

- `outputs/phase15/execution_sandbox_summary_latest.json`
- `outputs/phase15/execution_sandbox_report_2026-04-24.md`
- `outputs/phase15/target_weights_2026-04-24.json`
- `outputs/phase15/portfolio_delta_2026-04-24.json`
- `outputs/phase15/hypothetical_orders_2026-04-24.csv`

Replay and audit:

- `outputs/replay/real/weekly_replay.csv`
- `outputs/replay/real/numerical_health_summary.json`
- `outputs/replay/synthetic/weekly_replay.csv`
- `outputs/audit/state_stress_replay/behavior_audit_summary.csv`

These files are the auditable outputs that an operator or auditor should inspect.

## Test Matrix

Use the test suite to verify the contract before making operational claims.

| Area | Representative tests | What they protect |
|---|---|---|
| PIT and strict evidence | `tests/test_pit_contract.py`, `tests/test_publication_proof.py`, `tests/test_strict_evidence.py`, `tests/test_strict_real_pipeline.py` | No hindsight-adjusted data and no false strict eligibility |
| Backfill and store separation | `tests/test_backfill_validation.py`, `tests/test_backfill_store_separation.py`, `tests/test_controlled_backfill_live_contracts.py` | Degraded and strict stores remain separated |
| Ops status and publishing | `tests/test_phase14_ops_status.py`, `tests/test_phase14_publishing.py`, `tests/test_phase14_revision_audit.py` | Status model and publication outputs stay consistent |
| Phase 15 sandbox | `tests/test_phase15_sandbox_report.py`, `tests/test_portfolio_signal_gate.py`, `tests/test_order_router_simulation.py` | Paper-only sandbox never becomes broker submission |
| Micro and numerical health | `tests/test_micro_point_in_time.py`, `tests/test_micro_iir_freeze.py`, `tests/test_numerical_health.py`, `tests/test_covariance.py` | NaN handling, freeze behavior, and numerical stability |
| Replay and interpretation | `tests/test_real_replay.py`, `tests/test_replay_tables.py`, `tests/test_live_interpretability.py` | Historical replay remains auditable |

Useful validation command:

```bash
pytest -q
```

## Common Failure Modes

- `proof_strict_eligible=false`: publication evidence arrived after the SLA cutoff or cannot be machine verified.
- `current_status=BLOCK`: one or more sources, contracts, or freshness checks failed.
- `orders_count=0`: degraded signals, frozen micro state, or incomplete signal tuple.
- `h_t=null` or `rho_t=null`: micro layer is unavailable, so Phase 15 must not produce orders.
- `pit_engine not provided`: the micro layer cannot run in strict PIT mode.
- stale source warnings: the weekly report should be treated as degraded or blocked, not as executable output.

If a failure mode appears, inspect the upstream JSON or Markdown artifact first.
Do not patch the artifact by hand to make the status look better.

## What Not To Do

- Do not interpret sandbox output as a live trading instruction.
- Do not connect this repository to a broker API.
- Do not add account authentication for Phase 15.
- Do not claim live readiness from `degraded_backfill`.
- Do not claim strict recovery without machine-verifiable before-SLA proof.
- Do not manually edit outputs to force a passing status.
- Do not use backward-adjusted historical prices where the PIT contract is required.
- Do not change the model windows, half-lives, thresholds, or weights without an explicit spec change.

## FAQ / Troubleshooting

Why is the system still degraded?

Because the validation path can pass while the strict PIT publication proof still fails. For `2026-04-24`, the correct result is `degraded_backfill`.

Why are there zero orders in Phase 15?

Because the signal is degraded and the micro layer is frozen. Phase 15 intentionally emits no orders in this state.

Can I manually fix the JSON to unlock strict?

No. Strict eligibility comes from proof, not from editing files.

Can I submit the Phase 15 output to a broker?

No. The sandbox is paper-only and broker submission is disabled.

Where should I start when something looks wrong?

Start with `outputs/phase14/ops_status_summary.json`, then check the weekly report and the Phase 15 summary.

## Future Operator UI / Automation Roadmap

The repository currently supports command-line execution and generated artifacts.
Future operator UX items may include:

- scheduled cron orchestration
- email digest delivery
- dashboard views for weekly ops status
- richer operator approval workflows
- automated broker-adjacent handoff interfaces for paper-only review

Treat these as roadmap items unless the repository contains explicit scripts and configs that implement them.

The current operational truth remains:

- Phase 14 publishes auditable status artifacts
- Phase 15 remains paper-only
- no broker submission is allowed
- degraded backfill is the correct state when strict PIT proof is missing
