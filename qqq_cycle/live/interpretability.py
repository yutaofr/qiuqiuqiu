"""Structured interpretability snapshot for one live weekly run.

`build_snapshot()` assembles an `InterpretabilitySnapshot` from outputs that are
already present in the live run path — no recomputation, no new math.
`snapshot_to_dict()` converts it to a JSON-serializable dict for persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qqq_cycle.backtest.diagnostics import _semantic_label
from qqq_cycle.config import ModelConfig
from qqq_cycle.live.freshness import FreshnessRecord
from qqq_cycle.live.state_io import LiveState
from qqq_cycle.pipeline import PipelineResult


@dataclass(frozen=True)
class InterpretabilitySnapshot:
    """Auditable weekly interpretability object for one live run.

    All fields are derived from data already computed in ``LiveRuntime.run_week()``
    — this is a structured view, not a recomputation.
    """

    week_end: str

    # Execution tier (top-level resolution of signal + freshness state)
    execution_tier: str  # signal_invalid | signal_valid_not_executable | signal_valid_executable

    # State plane: current position in (H, I) space
    H_t: float | None
    I_t: float | None
    k_hat_t: int | None
    state_label: str | None
    state_probabilities: dict[str, float] | None   # {"0": p0, "1": p1, ...}
    centroids: list[list[float]] | None            # shape K×2

    # Factor attribution: raw factor values driving H and I
    factor_attribution: dict[str, float | None]   # L, T, P, E (H inputs)

    # Stress attribution: displacement, acceleration, final stress
    stress_attribution: dict[str, float | None]   # d, a, g_raw, g_stress, s_t

    # Drift metrics with threshold bands
    drift_metrics: dict[str, float | None]         # drift_probe_raw, drift_flag, threshold_lo, threshold_hi

    # Module health from live state bookkeeping
    health_metrics: dict[str, Any]                 # warmup_count, heal_count, breaker_active, h_t_lead_prev

    # Per-source freshness flags
    pollution_flags: dict[str, Any]                # {<source>_fresh: bool, ...} + stale_sources list


def build_snapshot(
    *,
    week_end: str,
    pipeline_result: PipelineResult,
    freshness: list[FreshnessRecord],
    execution_state: str,
    execution_permitted: bool,
    signal_valid_but_not_executable: bool,
    live_state: LiveState,
    config: ModelConfig,
) -> InterpretabilitySnapshot:
    """Assemble an InterpretabilitySnapshot from live run outputs.

    All inputs come from values already computed inside ``LiveRuntime.run_week()``.
    No additional data access or signal recomputation is performed.
    """
    interp = pipeline_result.interpretability or {}

    # --- Execution tier ---
    if execution_permitted:
        execution_tier = "signal_valid_executable"
    elif signal_valid_but_not_executable:
        execution_tier = "signal_valid_not_executable"
    else:
        execution_tier = "signal_invalid"

    # --- State plane ---
    H_t: float | None = interp.get("H")
    I_t: float | None = interp.get("I")
    k_hat_t: int | None = interp.get("k_hat_t")

    state_label: str | None = None
    state_probabilities: dict[str, float] | None = None
    centroids: list[list[float]] | None = None

    proto = live_state.proto
    if proto is not None and k_hat_t is not None:
        state_label = _semantic_label(proto.centroids, k_hat_t)
        centroids = proto.centroids.tolist()

    p_t: list[float] | None = interp.get("p_t")
    if p_t is not None:
        state_probabilities = {str(i): float(p) for i, p in enumerate(p_t)}

    # --- Factor attribution ---
    factor_attribution: dict[str, float | None] = {
        "L": interp.get("L"),
        "T": interp.get("T"),
        "P": interp.get("P"),
        "E": interp.get("E"),
    }

    # --- Stress attribution ---
    stress_attribution: dict[str, float | None] = {
        "displacement_d": interp.get("d"),
        "acceleration_a": interp.get("a"),
        "g_raw": interp.get("g_raw"),
        "g_stress": interp.get("g_stress"),
        "s_t": interp.get("s"),
    }

    # --- Drift metrics ---
    drift_metrics: dict[str, float | None] = {
        "drift_probe_raw": interp.get("drift_probe_raw"),
        "drift_flag": float(interp.get("drift_flag") or 0),
        "threshold_lo": float(config.drift.theta_lo),
        "threshold_hi": float(config.drift.theta_hi),
    }

    # --- Health metrics ---
    health_metrics: dict[str, Any] = {
        "warmup_count": live_state.warmup_count,
        "heal_count": live_state.heal_count,
        "breaker_active": live_state.breaker_active,
        "h_t_lead_prev": live_state.h_t_lead_prev,
    }

    # --- Pollution flags ---
    stale: list[str] = []
    flags: dict[str, Any] = {}
    for rec in freshness:
        key = f"{rec.source_label}_fresh"
        flags[key] = rec.fresh_enough
        if not rec.fresh_enough:
            stale.append(rec.source_label)
    flags["stale_sources"] = stale
    flags["execution_tier"] = execution_tier

    return InterpretabilitySnapshot(
        week_end=week_end,
        execution_tier=execution_tier,
        H_t=H_t,
        I_t=I_t,
        k_hat_t=k_hat_t,
        state_label=state_label,
        state_probabilities=state_probabilities,
        centroids=centroids,
        factor_attribution=factor_attribution,
        stress_attribution=stress_attribution,
        drift_metrics=drift_metrics,
        health_metrics=health_metrics,
        pollution_flags=flags,
    )


def snapshot_to_dict(snap: InterpretabilitySnapshot) -> dict[str, Any]:
    """Convert an InterpretabilitySnapshot to a JSON-serializable dict."""
    return {
        "week_end": snap.week_end,
        "execution_tier": snap.execution_tier,
        "state_plane": {
            "H_t": snap.H_t,
            "I_t": snap.I_t,
            "k_hat_t": snap.k_hat_t,
            "state_label": snap.state_label,
            "state_probabilities": snap.state_probabilities,
            "centroids": snap.centroids,
        },
        "factor_attribution": snap.factor_attribution,
        "stress_attribution": snap.stress_attribution,
        "drift_metrics": snap.drift_metrics,
        "health_metrics": snap.health_metrics,
        "pollution_flags": snap.pollution_flags,
    }
