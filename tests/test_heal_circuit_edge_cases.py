"""T9: Heal circuit edge case tests for the inline IIR envelope in pipeline.py.

Three sequences exercising the state machine at _HEAL_CIRCUIT_WEEKS = 3:
  (a) high -> 2-low -> high  : no early clear (heal_count resets to 0 on high)
  (b) high -> 3-low          : circuit fires at exactly week 3 (envelope reset)
  (c) NaN gap in h_t         : missing values are skipped; heal_count not advanced

All tests drive run_pipeline() directly so the exact IIR code path is tested.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from qqq_cycle.core.state_layer import compute_state_layer
from qqq_cycle.pipeline import (
    MODE_DEGRADED,
    MODE_STRICT,
    MODE_WARMUP,
    PipelineContracts,
    run_pipeline,
)
from tests.fixtures.strict_pipeline_fixture import make_strict_macro_inputs


# ── module-level fixture (computed once) ─────────────────────────────────────
# Compute the actual warmup length from state_layer without running the full
# pipeline. The covariance gate fires after 260 consecutive finite I (theta)
# values; the row AT which the 260th finite update occurs is still warmup,
# so the first post-warmup row is one beyond that.

_INPUTS = make_strict_macro_inputs()
_I_SERIES = compute_state_layer(_INPUTS)["I"]
_FINITE_CUMCOUNT = np.isfinite(_I_SERIES.values).cumsum()
# Index of the row where cumulative finite count first reaches 260 (still warmup).
_WARMUP_LAST_IDX = int(np.argmax(_FINITE_CUMCOUNT >= 260))
# First post-warmup row index.
_FIRST_POST_WARMUP = _WARMUP_LAST_IDX + 1


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_contracts_with_h_t(h_t_by_offset: dict) -> PipelineContracts:
    """Build PipelineContracts with specific h_t values at post-warmup offsets.

    h_t_by_offset: {offset_from_first_strict_row: value | float('nan')}
    Warmup entries (before _FIRST_POST_WARMUP) are NaN.
    Other post-warmup entries default to 0.30.
    """
    index = _INPUTS.index
    n = len(index)
    values = [float("nan")] * n
    for i in range(_FIRST_POST_WARMUP, n):
        offset = i - _FIRST_POST_WARMUP
        values[i] = h_t_by_offset.get(offset, 0.30)
    series = pd.Series(values, index=index)
    return PipelineContracts(
        weekly_h_t=series,
        pit_engine_available=True,
        constituents_available=True,
        weights_available=True,
    )


def _post_warmup_results(results):
    return [r for r in results if r.mode != MODE_WARMUP]


# ── test (a): high -> 2-low -> high  →  no early envelope clear ───────────────

def test_heal_circuit_no_early_clear():
    """2 consecutive sub-threshold weeks do NOT fire the circuit.

    Sequence at post-warmup offsets 0-3:
        offset 0: h_t=0.70  (high, envelope=0.70)
        offset 1: h_t=0.05  (low, heal_count=1)
        offset 2: h_t=0.05  (low, heal_count=2 — needs 3 to fire)
        offset 3: h_t=0.70  (high again, heal_count resets to 0)

    At offsets 1 and 2 the envelope is still carried from 0.70 via IIR decay,
    so rho_t must be non-null (IIR state kept row in STRICT, not DEGRADED).
    """
    contracts = _make_contracts_with_h_t({0: 0.70, 1: 0.05, 2: 0.05, 3: 0.70})
    results = run_pipeline(_INPUTS, contracts=contracts)
    post_wu = _post_warmup_results(results)
    assert len(post_wu) >= 4, "fixture must produce at least 4 post-warmup rows"

    r0, r1, r2, r3 = post_wu[:4]

    assert r0.mode == MODE_STRICT
    assert r0.h_t == pytest.approx(0.70, abs=1e-9)

    # Offsets 1 and 2: still strict (h_t_raw non-NaN), rho_t from IIR envelope.
    for r in (r1, r2):
        assert r.mode == MODE_STRICT, f"expected STRICT for low offset, got {r.mode}"
        assert r.h_t == pytest.approx(0.05, abs=1e-9)
        assert r.rho_t is not None, "rho_t must be non-null; IIR envelope carried"

    # Offset 3: high again, heal_count resets.
    assert r3.mode == MODE_STRICT
    assert r3.h_t == pytest.approx(0.70, abs=1e-9)
    assert r3.rho_t is not None


# ── test (b): high -> 3-low  →  circuit fires exactly at week 3 ──────────────

def test_heal_circuit_fires_at_exactly_3():
    """Circuit fires at exactly heal_count == 3, resetting h_t_lead to h_t_raw.

    Sequence:
        offset 0: h_t=0.80  (high, envelope=0.80)
        offset 1: h_t=0.03  (low, heal_count=1)
        offset 2: h_t=0.03  (low, heal_count=2)
        offset 3: h_t=0.03  (low, heal_count=3 → circuit fires → reset count=0)

    All four rows must remain STRICT (h_t_raw is non-NaN throughout).
    After the circuit fires at offset 3, h_t stays 0.03 (raw value stored in result).
    """
    contracts = _make_contracts_with_h_t({0: 0.80, 1: 0.03, 2: 0.03, 3: 0.03})
    results = run_pipeline(_INPUTS, contracts=contracts)
    post_wu = _post_warmup_results(results)
    assert len(post_wu) >= 4

    r0, r1, r2, r3 = post_wu[:4]

    for i, r in enumerate((r0, r1, r2, r3)):
        assert r.mode == MODE_STRICT, f"offset {i}: expected STRICT, got {r.mode}"
        assert r.h_t is not None, f"offset {i}: h_t must be non-null"
        assert r.rho_t is not None, f"offset {i}: rho_t must be non-null"

    # h_t stores the raw value (not the IIR lead).
    assert r0.h_t == pytest.approx(0.80, abs=1e-9)
    assert r1.h_t == pytest.approx(0.03, abs=1e-9)
    assert r2.h_t == pytest.approx(0.03, abs=1e-9)
    assert r3.h_t == pytest.approx(0.03, abs=1e-9)


# ── test (c): NaN gap does not advance heal_count ────────────────────────────

def test_heal_circuit_nan_gap_does_not_advance():
    """A NaN entry in weekly_h_t skips the IIR step; heal_count must not increment.

    Sequence:
        offset 0: h_t=0.70  (high, envelope=0.70, heal_count=0)
        offset 1: h_t=0.05  (low, heal_count=1)
        offset 2: h_t=NaN   (degraded row — IIR skipped entirely, heal_count stays 1)
        offset 3: h_t=0.05  (low, heal_count=2 — circuit NOT yet fired at 2)
        offset 4: h_t=0.05  (low, heal_count=3 → circuit fires)

    Offset 2 must be MODE_DEGRADED (no h_t).
    Offsets 3 and 4 must be MODE_STRICT with non-null rho_t.
    """
    contracts = _make_contracts_with_h_t(
        {0: 0.70, 1: 0.05, 2: float("nan"), 3: 0.05, 4: 0.05}
    )
    results = run_pipeline(_INPUTS, contracts=contracts)
    post_wu = _post_warmup_results(results)
    assert len(post_wu) >= 5

    r0, r1, r2, r3, r4 = post_wu[:5]

    # offset 0: strict, high
    assert r0.mode == MODE_STRICT
    assert r0.h_t == pytest.approx(0.70, abs=1e-9)

    # offset 1: strict, low — envelope carried from 0.70
    assert r1.mode == MODE_STRICT
    assert r1.h_t == pytest.approx(0.05, abs=1e-9)
    assert r1.rho_t is not None

    # offset 2: NaN → degraded, no h_t/rho_t
    assert r2.mode == MODE_DEGRADED, (
        f"expected DEGRADED for NaN row, got {r2.mode}"
    )
    assert r2.h_t is None
    assert r2.rho_t is None

    # offset 3: strict again; heal_count=2, circuit not yet fired
    assert r3.mode == MODE_STRICT
    assert r3.h_t == pytest.approx(0.05, abs=1e-9)
    assert r3.rho_t is not None

    # offset 4: strict; heal_count=3, circuit fires (envelope reset to 0.05)
    assert r4.mode == MODE_STRICT
    assert r4.h_t == pytest.approx(0.05, abs=1e-9)
    assert r4.rho_t is not None
