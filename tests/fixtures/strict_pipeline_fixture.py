"""Strict-mode pipeline fixture for Phase 6 integration testing.

Provides synthetic macro inputs and PipelineContracts with a pre-computed
weekly_h_t series that covers the full date index:
    - NaN for the first warmup_weeks entries (warmup rows must not consume h_t)
    - U(0.1, 0.6) values for post-warmup entries (activates strict mode)

This proves routing correctness without requiring real PIT/constituent/weight data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qqq_cycle.backtest.diagnostics import synthetic_replay_inputs
from qqq_cycle.config import load_config
from qqq_cycle.pipeline import PipelineContracts, PipelineResult, run_pipeline

_RNG_SEED = 20260428
_WARMUP_WEEKS = 260  # Must match model_v22.yaml warmup_weeks


def make_strict_macro_inputs() -> pd.DataFrame:
    """Synthetic weekly macro series spanning 2000-2025.

    Delegates to the existing deterministic fixture generator used by the
    replay audit tests. RNG seed is fixed in synthetic_replay_inputs().
    """
    return synthetic_replay_inputs()


def make_strict_contracts(weekly_index: pd.DatetimeIndex) -> PipelineContracts:
    """Build PipelineContracts with full-index weekly_h_t.

    weekly_h_t layout:
        - First _WARMUP_WEEKS rows: NaN (warmup period)
        - Remaining rows: uniform random in [0.10, 0.60] (post-warmup signal)

    Full-index coverage simultaneously validates:
        (a) Warmup rows silently skip NaN h_t without entering strict mode
        (b) Post-warmup rows switch to strict mode once h_t is non-NaN
    """
    rng = np.random.default_rng(_RNG_SEED)
    n = len(weekly_index)
    values = np.full(n, np.nan)
    post_warmup_count = max(0, n - _WARMUP_WEEKS)
    values[_WARMUP_WEEKS:] = rng.uniform(0.10, 0.60, size=post_warmup_count)
    h_t_series = pd.Series(values, index=weekly_index, name="h_t")
    return PipelineContracts(
        weekly_h_t=h_t_series,
        pit_engine_available=True,
        constituents_available=True,
        weights_available=True,
    )


def run_strict_fixture() -> list[PipelineResult]:
    """Run pipeline with synthetic inputs and strict contracts.

    Returns:
        List of PipelineResult covering 2000-2025 weekly dates.
        Post-warmup rows should have mode="strict" with non-null h_t and rho_t.
    """
    config = load_config()
    inputs = make_strict_macro_inputs()
    contracts = make_strict_contracts(inputs.index)
    return run_pipeline(inputs, contracts=contracts, config=config)
