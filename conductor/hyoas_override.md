# Implementing HYOAS CSV Override for Real Replay

## Objective
Create support for a licensed external long-history HYOAS input file for the `BAMLH0A0HYM2` series due to FRED licensing changes restricting data to 3 years. This will unblock the real replay process without fetching or fabricating proprietary history.

## Scope & Impact
- **Impacted Files**:
  - `qqq_cycle/data_contracts/hyoas_override.py` (New)
  - `qqq_cycle/backtest/real_replay.py`
  - `tests/test_hyoas_override.py` (New)
  - `tests/test_real_replay.py`
  - `data/licensed/hyoas_bamlh0a0hym2.csv` (Sample Template)
- **Scope**: Adding strict data ingestion and validation for the HYOAS override. Wiring the override into the replay generation. Adding unit and integration tests.

## Implementation Steps

### 1. Add Rigid CSV Contract (`qqq_cycle/data_contracts/hyoas_override.py`)
Create a new data contract and loader store that validates:
- **Required Columns**: `trade_date`, `value`, `source_name`, `source_timestamp`, `license_tag`.
- **Optional Columns**: `notes`.
- **Validation Rules**:
  - `trade_date`: required, parseable, daily, strictly increasing, no duplicates.
  - `value`: required, numeric, finite.
  - Minimum coverage check (must be sufficient for state/stress warmup, i.e., ≥ 525 rows).
  - Explicit non-null strings for `source_name`, `source_timestamp`, `license_tag`.
  - Reject files failing any rule.

### 2. Implement Manifest-Backed Loader
- The loader will generate a manifest alongside the accepted file containing: `source_name`, `source_timestamp`, `min_date`, `max_date`, `row_count`, `sha256` (optional/hash representation), `coverage_ok`, and `reason_if_rejected` if failed.

### 3. Wire Override into Real Replay (`qqq_cycle/backtest/real_replay.py`)
- Try FRED first. If history is insufficient (e.g., < 525 rows), try the licensed local override CSV via the new contract.
- If override passes validation, use it and set `hyoas_source = "licensed_csv_override"`.
- Record the manifest details inside the overall `real_replay` manifest.
- If the override is missing or invalid, explicitly degrade the replay. Do not substitute or change model math.

### 4. Create Template CSV (`data/licensed/hyoas_bamlh0a0hym2.csv`)
Save the exact provided CSV template to be used by the system when testing/running if the real data is provided by the user.

### 5. Testing
- Create `tests/test_hyoas_override.py` to verify:
  - Accepts compliant CSV.
  - Rejects duplicates, non-numeric values, missing required fields, and insufficient coverage.
- Update `tests/test_real_replay.py` to verify:
  - Real replay succeeds when a compliant HYOAS override is present.
  - Real replay gracefully degrades when the override is absent or invalid.

## Verification & Rollback
- Run the full pytest suite (`pytest tests/`).
- Execute a smoke test using the existing `scripts/run_real_replay.py` with the template data (even if insufficient, it should gracefully degrade or we can mock a sufficient version for testing).
- Rollback is simply reverting the git changes to `real_replay.py`.