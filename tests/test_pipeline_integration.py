"""Phase 6 integration tests: strict fixture path and degraded real path.

Six required tests:
    1. test_result_schema          — PipelineResult has all required fields
    2. test_warmup_gate_enforced   — warmup rows have all outputs null (incl. s_t)
    3. test_strict_fixture_full_tuple — strict rows have non-null h_t and rho_t
    4. test_degraded_real_h_t_rho_t_null — real replay with no contracts: h_t/rho_t null
    5. test_strict_to_degraded_routing — removing weekly_h_t triggers degraded, not crash
    6. test_no_silent_fallback     — every degraded row has non-empty degraded_reason
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qqq_cycle.core.interpretability import InterpretabilityRecord
from qqq_cycle.config import load_config
from qqq_cycle.data_contracts.constituents import PITConstituentSnapshot
from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError, PITAdjustmentEngine
from qqq_cycle.pipeline import (
    MODE_DEGRADED,
    MODE_STRICT,
    MODE_WARMUP,
    PipelineContracts,
    PipelineResult,
    _build_audit_interpretability,
    _compute_daily_micro_frame,
    _compute_weekly_h_t_from_stores,
    results_to_frame,
    run_pipeline,
)
from tests.fixtures.strict_pipeline_fixture import (
    make_strict_contracts,
    make_strict_macro_inputs,
    run_strict_fixture,
)

_REAL_STAGING_CSV = Path("cache/real_replay/staging/weekly_inputs.csv")


# ---------------------------------------------------------------------------
# Test 1: result schema
# ---------------------------------------------------------------------------

def test_result_schema():
    """PipelineResult has all required fields with correct types."""
    r = PipelineResult(
        week_end="2024-01-05",
        k_hat_t=None,
        p_t=None,
        s_t=None,
        h_t=None,
        rho_t=None,
        I_t=None,
        interpretability=None,
        mode=MODE_WARMUP,
        degraded_reason=None,
        strict_contracts_satisfied=None,
    )
    required_fields = [
        "week_end", "k_hat_t", "p_t", "s_t", "h_t", "rho_t", "I_t",
        "interpretability", "mode", "degraded_reason", "strict_contracts_satisfied",
        "backfill_mode", "micro_state_frozen", "micro_envelope_internal_state",
        "micro_breaker_internal_state", "micro_rho_update_state",
        "contract_source", "strict_gate_passed",
    ]
    for field in required_fields:
        assert hasattr(r, field), f"PipelineResult missing field: {field}"

    assert r.mode == MODE_WARMUP
    d = r.to_dict()
    assert d["mode"] == MODE_WARMUP
    assert set(required_fields) == set(d.keys())

    # results_to_frame must produce a DataFrame with the same columns
    frame = results_to_frame([r])
    assert set(required_fields) == set(frame.columns)


# ---------------------------------------------------------------------------
# Test 2: warmup gate enforced
# ---------------------------------------------------------------------------

def test_warmup_gate_enforced():
    """First ≥260 weeks emit mode='warmup' with ALL output fields null.

    Specifically: s_t must be null during warmup — it must NOT be read from
    stress_frame before cov.is_warm(). This is the key divergence from
    diagnostics.py which does output s_t during warmup rows.
    """
    results = run_strict_fixture()
    warmup_rows = [r for r in results if r.mode == MODE_WARMUP]
    assert len(warmup_rows) >= 260, (
        f"expected ≥260 warmup rows, got {len(warmup_rows)}"
    )
    for r in warmup_rows:
        assert r.k_hat_t is None, f"k_hat_t must be null in warmup row {r.week_end}"
        assert r.p_t is None, f"p_t must be null in warmup row {r.week_end}"
        assert r.s_t is None, (
            f"s_t must be null in warmup row {r.week_end}, got {r.s_t}. "
            "This fires if pipeline reads stress_frame before cov.is_warm()."
        )
        assert r.h_t is None, f"h_t must be null in warmup row {r.week_end}"
        assert r.rho_t is None, f"rho_t must be null in warmup row {r.week_end}"
        assert r.I_t is None, f"I_t must be null in warmup row {r.week_end}"
        assert r.strict_contracts_satisfied is None, (
            f"strict_contracts_satisfied must be None in warmup row {r.week_end}"
        )
        assert r.degraded_reason is None, (
            f"degraded_reason must be None in warmup row {r.week_end}"
        )


# ---------------------------------------------------------------------------
# Test 3: strict fixture full tuple
# ---------------------------------------------------------------------------

def test_strict_fixture_full_tuple():
    """Strict mode rows have the full model-spec output tuple."""
    results = run_strict_fixture()
    strict_rows = [r for r in results if r.mode == MODE_STRICT]
    assert len(strict_rows) > 0, "no strict rows produced — fixture or routing broken"

    for r in strict_rows:
        assert r.h_t is not None, f"h_t must be non-null in strict row {r.week_end}"
        assert r.rho_t is not None, f"rho_t must be non-null in strict row {r.week_end}"
        assert isinstance(r.I_t, InterpretabilityRecord), (
            f"I_t must be InterpretabilityRecord in strict row {r.week_end}"
        )
        assert r.k_hat_t is not None, f"k_hat_t must be non-null in strict row {r.week_end}"
        assert r.s_t is not None, f"s_t must be non-null in strict row {r.week_end}"
        assert r.degraded_reason is None, (
            f"degraded_reason must be None in strict row {r.week_end}"
        )
        assert 0.0 <= r.h_t <= 1.0, f"h_t={r.h_t} out of [0,1] in row {r.week_end}"
        assert 0.0 <= r.rho_t <= 1.0, f"rho_t={r.rho_t} out of [0,1] in row {r.week_end}"


def test_strict_fixture_serializes_audit_interpretability_object():
    """Strict result serialization includes auditable A/C/D/H interpretability."""
    strict_row = next(r for r in run_strict_fixture() if r.mode == MODE_STRICT)

    serialized = strict_row.to_dict()

    assert serialized["I_t"] is not None
    assert set(serialized["I_t"]) == {"A_t", "C_t", "D_t", "H_t"}
    assert set(serialized["I_t"]["A_t"]) == {
        "H_components",
        "I_components",
        "stress_components",
        "micro_components",
        "rho_components",
    }
    assert set(serialized["I_t"]["C_t"]) == {
        "c_rule",
        "c_const",
        "c_data",
        "c_micro",
        "c_drift",
    }
    assert set(serialized["I_t"]["D_t"]) == {
        "d_state",
        "d_stress",
        "d_frag",
        "d_abs",
    }
    assert set(serialized["I_t"]["H_t"]) == {
        "h_macro",
        "h_exo",
        "h_micro",
        "h_state",
    }


# ---------------------------------------------------------------------------
# Test 4: degraded real — h_t and rho_t null
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _REAL_STAGING_CSV.exists(),
    reason=f"real staging CSV not found: {_REAL_STAGING_CSV}",
)
def test_degraded_real_h_t_rho_t_null():
    """Real macro replay with contracts=None yields h_t=rho_t=null post-warmup."""
    inputs = pd.read_csv(_REAL_STAGING_CSV, index_col=0, parse_dates=True)
    inputs.index = pd.to_datetime(inputs.index)
    results = run_pipeline(inputs, contracts=None)

    post_warmup = [r for r in results if r.mode != MODE_WARMUP]
    assert len(post_warmup) > 0, "no post-warmup rows in real replay"

    for r in post_warmup:
        assert r.h_t is None, (
            f"h_t must be null in degraded row {r.week_end}, got {r.h_t}"
        )
        assert r.rho_t is None, (
            f"rho_t must be null in degraded row {r.week_end}, got {r.rho_t}"
        )

    # State/stress must still run post-warmup.
    assert any(r.k_hat_t is not None for r in post_warmup), (
        "state engine must produce k_hat_t on at least some post-warmup rows"
    )
    assert any(r.s_t is not None for r in post_warmup), (
        "stress engine must produce s_t on at least some post-warmup rows"
    )


# ---------------------------------------------------------------------------
# Test 5: strict-to-degraded routing
# ---------------------------------------------------------------------------

def test_strict_to_degraded_routing():
    """Removing weekly_h_t from contracts triggers degraded mode — not a crash."""
    inputs = make_strict_macro_inputs()
    # Contracts with all boolean flags True but weekly_h_t=None.
    contracts_no_h = PipelineContracts(
        weekly_h_t=None,
        pit_engine_available=True,
        constituents_available=True,
        weights_available=True,
    )
    results = run_pipeline(inputs, contracts=contracts_no_h)

    post_warmup = [r for r in results if r.mode != MODE_WARMUP]
    assert len(post_warmup) > 0, "no post-warmup rows produced"

    strict_rows = [r for r in post_warmup if r.mode == MODE_STRICT]
    assert len(strict_rows) == 0, (
        f"{len(strict_rows)} rows are still strict after removing weekly_h_t — routing broken"
    )

    for r in post_warmup:
        assert r.mode == MODE_DEGRADED, (
            f"expected MODE_DEGRADED but got {r.mode!r} for row {r.week_end}"
        )
        assert r.h_t is None, f"h_t must be null in degraded row {r.week_end}"
        assert r.rho_t is None, f"rho_t must be null in degraded row {r.week_end}"
        assert r.strict_contracts_satisfied is False, (
            f"strict_contracts_satisfied must be False in degraded row {r.week_end}"
        )


# ---------------------------------------------------------------------------
# Test 6: no silent fallback
# ---------------------------------------------------------------------------

def test_no_silent_fallback():
    """Every degraded row has a non-empty degraded_reason string.

    Every non-degraded row must have degraded_reason=None.
    """
    inputs = make_strict_macro_inputs()
    results = run_pipeline(inputs, contracts=None)

    for r in results:
        if r.mode == MODE_DEGRADED:
            assert r.degraded_reason is not None, (
                f"degraded_reason must not be None in degraded row {r.week_end}"
            )
            assert len(r.degraded_reason) > 0, (
                f"degraded_reason must be non-empty in degraded row {r.week_end}"
            )
        else:
            assert r.degraded_reason is None, (
                f"degraded_reason must be None in {r.mode} row {r.week_end}, "
                f"got {r.degraded_reason!r}"
            )


class _NonStrictPITEngine(PITAdjustmentEngine):
    asof_semantics = "eod_same_day"

    def get_adjusted_window(self, ticker, end_date, window, asof):  # pragma: no cover
        raise AssertionError("non-strict PIT engine must not be used for micro strict mode")


class _RecordingStrictPITEngine(PITAdjustmentEngine):
    asof_semantics = "strict_pit"

    def __init__(self, *, missing_60: set[str] | None = None) -> None:
        super().__init__()
        self.missing_60 = missing_60 or set()
        self.calls: list[tuple[str, pd.Timestamp, int, pd.Timestamp]] = []

    def get_adjusted_window(
        self, ticker: str, end_date: pd.Timestamp, window: int, asof: pd.Timestamp
    ) -> pd.Series:
        self.calls.append((ticker, pd.Timestamp(end_date), window, pd.Timestamp(asof)))
        if window == 60 and ticker in self.missing_60:
            raise DataNotAvailableError(f"missing 60-day PIT window for {ticker}")
        idx = pd.bdate_range(end=pd.Timestamp(end_date), periods=window)
        if ticker == "C" and window == 20:
            values = [100.0] * (window - 1) + [80.0]
        else:
            values = [100.0] * (window - 1) + [120.0]
        return pd.Series(values, index=idx, dtype=float)


class _StableConstituentStore:
    def __init__(self, members: frozenset[str]) -> None:
        self.members = members

    def get_snapshot(self, trade_date: pd.Timestamp, asof: pd.Timestamp) -> PITConstituentSnapshot:
        return PITConstituentSnapshot(pd.Timestamp(trade_date), self.members, pd.Timestamp(asof))


class _YoungMemberConstituentStore:
    def __init__(self, *, young_start: pd.Timestamp) -> None:
        self.young_start = pd.Timestamp(young_start)

    def get_snapshot(self, trade_date: pd.Timestamp, asof: pd.Timestamp) -> PITConstituentSnapshot:
        members = {"A", "C"}
        if pd.Timestamp(trade_date) >= self.young_start:
            members.add("B")
        return PITConstituentSnapshot(
            pd.Timestamp(trade_date), frozenset(members), pd.Timestamp(asof)
        )


class _StaticWeightStore:
    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights

    def get_weights(self, trade_date: pd.Timestamp, asof: pd.Timestamp) -> dict[str, float]:
        del trade_date, asof
        return dict(self.weights)


class _FlipWeightStore:
    def __init__(self, *, flip_date: pd.Timestamp) -> None:
        self.flip_date = pd.Timestamp(flip_date)

    def get_weights(self, trade_date: pd.Timestamp, asof: pd.Timestamp) -> dict[str, float]:
        del asof
        if pd.Timestamp(trade_date) >= self.flip_date:
            return {"A": 0.0, "C": 1.0}
        return {"A": 1.0, "C": 0.0}


def test_store_path_rejects_non_strict_pit_engine() -> None:
    inputs = make_strict_macro_inputs()
    contracts = PipelineContracts(
        pit_engine=_NonStrictPITEngine(),
        constituent_store=_StableConstituentStore(frozenset({"A", "C"})),
        weight_store=_StaticWeightStore({"A": 0.6, "C": 0.4}),
    )

    results = run_pipeline(inputs, contracts=contracts)
    post_warmup = [r for r in results if r.mode != MODE_WARMUP]

    assert post_warmup
    assert all(r.mode == MODE_DEGRADED for r in post_warmup)
    assert all(r.h_t is None and r.rho_t is None for r in post_warmup)
    assert all(r.strict_contracts_satisfied is False for r in post_warmup)
    assert all("pit_engine" in (r.degraded_reason or "") for r in post_warmup)


def test_store_daily_micro_uses_mature_sets_not_present_members() -> None:
    weeks = pd.date_range("2024-01-05", periods=16, freq="W-FRI")
    young_start = weeks[-1] - pd.tseries.offsets.BDay(10)
    pit = _RecordingStrictPITEngine()
    contracts = PipelineContracts(
        pit_engine=pit,
        constituent_store=_YoungMemberConstituentStore(young_start=young_start),
        weight_store=_StaticWeightStore({"A": 0.45, "B": 0.10, "C": 0.45}),
    )

    daily = _compute_daily_micro_frame(pd.bdate_range(weeks[0], weeks[-1]), contracts)

    assert daily["b_tau"].notna().any()
    assert any(call[2] == 20 for call in pit.calls)
    assert any(call[2] == 60 for call in pit.calls)
    assert all(call[0] != "B" for call in pit.calls)


def test_store_daily_micro_uses_prior_day_smoothed_lagged_weights() -> None:
    days = pd.bdate_range("2024-01-02", periods=25)
    pit = _RecordingStrictPITEngine()
    contracts = PipelineContracts(
        pit_engine=pit,
        constituent_store=_StableConstituentStore(frozenset({"A", "C"})),
        weight_store=_FlipWeightStore(flip_date=days[-1]),
    )

    daily = _compute_daily_micro_frame(days, contracts)
    last_breadth = daily["b_tau"].dropna().iloc[-1]

    assert last_breadth < 0.1


def test_weekly_store_micro_fails_closed_when_any_mature_60d_window_missing() -> None:
    weeks = pd.date_range("2021-01-01", periods=180, freq="W-FRI")
    contracts = PipelineContracts(
        pit_engine=_RecordingStrictPITEngine(missing_60={"C"}),
        constituent_store=_StableConstituentStore(frozenset({"A", "C", "D"})),
        weight_store=_StaticWeightStore({"A": 0.34, "C": 0.33, "D": 0.33}),
    )

    h_t = _compute_weekly_h_t_from_stores(weeks, contracts, load_config())

    assert h_t.isna().all()


def test_audit_interpretability_marks_h_state_zero_when_covariance_unhealthy() -> None:
    week = pd.Timestamp("2024-01-05")
    state = pd.DataFrame(
        {"L": [1.0], "T": [2.0], "P": [3.0], "E": [0.5]},
        index=[week],
    )
    stress = pd.DataFrame({"g_raw": [0.1], "g_stress": [0.2]}, index=[week])
    drift = pd.DataFrame({"drift_probe_raw": [0.0]}, index=[week])

    record = _build_audit_interpretability(
        week,
        state,
        stress,
        drift,
        omega_t=0.6,
        s_t=0.4,
        n_t=0.2,
        h_t=0.7,
        h_t_available=True,
        rho_t_available=True,
        config=load_config(),
        state_ok=False,
    )

    assert record.H_t.h_state == 0
