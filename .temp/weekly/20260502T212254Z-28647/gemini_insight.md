# Weekly Digest Insight: 2026-05-01

**System Status:** `DEGRADED`
**Execution Status:** `DISALLOWED`

### Summary
The system is operating in **degraded mode** for the period ending 2026-05-01. The **strict gate failed** due to a `degraded_backfill_signal`, resulting in the suspension of all live execution and broker submissions.

### Key Metrics
*   **Orders Generated:** 0
*   **Signal Eligibility:** False
*   **Operational Mode:** Paper-only (Execution/Broker Submission blocked)
*   **Strict Gate:** Failed

### Root Cause
The transition to degraded mode was triggered by `phase14` failing the strict gate requirements. Operational safety protocols have locked the system to prevent execution until signal integrity is restored.
