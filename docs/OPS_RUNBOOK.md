# Phase 14 Ops Runbook

This document is static. Phase 14 scripts may reference it, but they must not rewrite it.

## 1. Weekly Cadence

### 1.1 Standard Operating Window

- Decision `week_end` is Friday.
- Operational SLA cutoff is Saturday `12:00` `America/New_York`.
- Before the cutoff, the current Friday snapshot is not overdue.
- After the cutoff, the current Friday snapshot is required for normal operations.

## 2. Snapshot And Freshness Issues

### 2.1 Missing Required Snapshot

1. Confirm the required `week_end` from the dynamic ops summary.
2. Check whether `outputs/phase14/history/` contains any snapshot for that week.
3. If the week is missing entirely, rerun publication only after confirming `outputs/live/live_run_summary.json` is current.
4. If live artifacts are also missing, rerun the live pipeline before republishing.

### 2.2 Late Snapshot Publication

1. Compare `published_at` to the SLA cutoff for the same `week_end`.
2. If publication happened after the cutoff, log the SLA miss.
3. Do not delete earlier history files; publish an additional immutable snapshot if a rerun is required.

### 2.3 Source Freshness Failures

1. Read the stale source list from `ops_status_summary`.
2. If the stale source has blocking level `block`, execution readiness must stay blocked until the source is refreshed.
3. If the stale source has blocking level `degrade`, keep the signal but mark the weekly package degraded.
4. After source refresh, rerun live publication and then rerun Phase 14 publishing and ops.

## 3. Signal And Execution Gates

### 3.1 Execution Degraded

1. Read the `degraded_reason` in the dynamic status summary.
2. Confirm whether the signal tuple is still valid (`k_hat_t`, `p_t`, `s_t` present).
3. Treat degraded execution as non-normal operations; do not classify the week as clean strict production.

### 3.2 Execution Blocked

1. Read the `execution_block_reason` in the dynamic status summary.
2. Resolve block-level freshness or contract failures before any weekly action.
3. Republish after the blocking cause is removed. Never overwrite immutable history.

### 3.3 Signal Invalid

1. If the signal tuple is incomplete, treat the week as invalid for operational review.
2. Confirm whether the snapshot is a warmup or malformed publication.
3. Rebuild upstream live artifacts before using any downstream Phase 14 outputs.

## 4. Revision Audit

### 4.1 Material Revisions

1. Review `revision_stability_detail.csv` for the flagged `week_end`.
2. Compare earliest vs latest snapshot fields and confirm which of:
   - `mode`
   - `k_hat_t`
   - `s_t`
   - `h_t`
   - `rho_t`
   triggered the material flag.
3. Preserve all immutable history files. Never collapse same-week reruns into a single file.

### 4.2 Same-Week Reruns

1. Same-week reruns are expected to accumulate as immutable files.
2. Use earliest/latest comparisons to explain why the public latest view changed.
3. Do not infer stability from `latest` pointers alone.

## 5. Regime Monitoring Review

### 5.1 Transition Events

1. Use `state_transition_matrix.csv` to inspect regime transitions from the latest-view history.
2. Use `state_duration_summary.csv` to review current and historical run lengths.
3. Use `event_response_summary.csv` to inspect `s_t`, `h_t`, and `rho_t` changes around transition weeks.

