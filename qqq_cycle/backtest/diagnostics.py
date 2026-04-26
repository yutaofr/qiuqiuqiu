"""Replay table generation for the audited state/stress slice."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qqq_cycle.core.covariance import RobustEWCov2D
from qqq_cycle.core.drift_probe import DriftProbe
from qqq_cycle.core.proto_online import (
    PrototypeState,
    initialize_prototypes_from_history,
    update_prototypes,
)
from qqq_cycle.core.state_layer import compute_state_layer
from qqq_cycle.core.stress_layer import compute_stress_layer

EVENT_WINDOWS: dict[str, tuple[str, str]] = {
    "2008_09_to_2009_06": ("2008-09-01", "2009-06-30"),
    "2020_02_to_2020_06": ("2020-02-01", "2020-06-30"),
    "2021_10_to_2022_03": ("2021-10-01", "2022-03-31"),
}

REPLAY_COLUMNS = [
    "week_end",
    "L_t",
    "T_t",
    "P_t",
    "E_t",
    "H_t",
    "I_t",
    "state_probs_json",
    "state_label",
    "d_t",
    "a_t",
    "g_t_raw",
    "g_t_stress",
    "s_t",
    "drift_probe_raw",
    "drift_flag",
    "maha",
    "huber_weight",
    "eigval_1",
    "eigval_2_raw",
    "eigval_2_reg",
    "condition_number_raw",
    "condition_number_reg",
    "eigval_2_was_floored",
    "state_ok",
    "is_warm",
]


@dataclass
class ReplayBundle:
    """Replay outputs and event-window extracts."""

    weekly: pd.DataFrame
    event_windows: dict[str, pd.DataFrame]


def synthetic_replay_inputs() -> pd.DataFrame:
    """Create deterministic fixture inputs spanning required event windows."""

    rng = np.random.default_rng(20260426)
    idx = pd.date_range("2000-01-07", "2025-12-26", freq="W-FRI")
    n = len(idx)
    trend = np.linspace(0.0, 1.0, n)
    shock = np.zeros(n)
    for start, end, amp in [
        ("2008-09-01", "2009-06-30", 2.0),
        ("2020-02-01", "2020-06-30", 1.5),
        ("2021-10-01", "2022-03-31", 1.2),
    ]:
        mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
        shock[mask] += amp
    qqq_returns = 0.002 + 0.015 * rng.normal(size=n) - 0.02 * shock
    qqq = 100 * np.exp(np.cumsum(qqq_returns))
    return pd.DataFrame(
        {
            "DFII10": 0.4 + 0.2 * np.sin(trend * 20) - 0.15 * shock + 0.03 * rng.normal(size=n),
            "DGS2": 2.0 + 0.5 * np.sin(trend * 12) + 0.25 * shock + 0.05 * rng.normal(size=n),
            "BAMLH0A0HYM2": 4.0 + 0.4 * np.cos(trend * 14) + 0.8 * shock + 0.08 * rng.normal(size=n),
            "NFCI": -0.25 + 0.2 * np.sin(trend * 17) + 0.5 * shock + 0.04 * rng.normal(size=n),
            "VIXCLS": 18.0 + 4.0 * np.sin(trend * 19) + 8.0 * shock + rng.normal(size=n),
            "AI_GPR": np.maximum(0, 50 + 15 * np.sin(trend * 18) + 20 * shock + 4 * rng.normal(size=n)),
            "USEPUINDXD": np.maximum(0, 90 + 20 * np.cos(trend * 15) + 25 * shock + 6 * rng.normal(size=n)),
            "QQQ": qqq,
        },
        index=idx,
    )


def _semantic_label(centroids: np.ndarray, cluster: int) -> str:
    order = np.argsort(centroids[:, 0])
    low = order[:2]
    mid = order[2]
    high = order[3:]
    labels: dict[int, str] = {}
    labels[int(low[np.argmin(centroids[low, 1])])] = "S1"
    labels[int(low[np.argmax(centroids[low, 1])])] = "S2"
    labels[int(mid)] = "S3"
    labels[int(high[np.argmax(centroids[high, 1])])] = "S4"
    labels[int(high[np.argmin(centroids[high, 1])])] = "S5"
    return labels[int(cluster)]


def _state_probabilities(theta: np.ndarray, proto: PrototypeState, cov_reg: np.ndarray) -> np.ndarray:
    inv_cov = np.linalg.inv(cov_reg)
    diffs = proto.centroids - theta
    d2 = np.einsum("ki,ij,kj->k", diffs, inv_cov, diffs)
    logits = -0.5 * d2
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    return probs / probs.sum()


def build_replay_bundle(inputs: pd.DataFrame) -> ReplayBundle:
    """Build weekly replay table and event-window extracts.

    This is a diagnostic replay only. It does not compute returns, micro-layer
    outputs, or risk-layer production results.
    """

    state = compute_state_layer(inputs)
    theta = state[["H", "I"]]
    finite_theta = theta.dropna()
    if len(finite_theta) < 265:
        raise RuntimeError("need at least 265 finite theta rows for replay")

    cov = RobustEWCov2D()
    cov_state = cov.initialize_from_history(finite_theta.iloc[:20].to_numpy())
    proto: PrototypeState | None = None
    proto_seed: list[np.ndarray] = []
    stress = compute_stress_layer(theta, state["E"]).frame
    drift = DriftProbe().compute(inputs)
    rows: list[dict[str, object]] = []

    for week_end, theta_row in theta.iterrows():
        x = theta_row.to_numpy(dtype=float)
        row: dict[str, object] = {
            "week_end": week_end.strftime("%Y-%m-%d"),
            "L_t": state.at[week_end, "L"] if week_end in state.index else np.nan,
            "T_t": state.at[week_end, "T"] if week_end in state.index else np.nan,
            "P_t": state.at[week_end, "P"] if week_end in state.index else np.nan,
            "E_t": state.at[week_end, "E"] if week_end in state.index else np.nan,
            "H_t": x[0] if np.isfinite(x[0]) else np.nan,
            "I_t": x[1] if np.isfinite(x[1]) else np.nan,
            "state_probs_json": np.nan,
            "state_label": "WARMUP",
            "d_t": stress.at[week_end, "d"] if week_end in stress.index else np.nan,
            "a_t": stress.at[week_end, "a"] if week_end in stress.index else np.nan,
            "g_t_raw": stress.at[week_end, "g_raw"] if week_end in stress.index else np.nan,
            "g_t_stress": stress.at[week_end, "g_stress"] if week_end in stress.index else np.nan,
            "s_t": stress.at[week_end, "s"] if week_end in stress.index else np.nan,
            "drift_probe_raw": drift.at[week_end, "drift_probe_raw"] if week_end in drift.index else np.nan,
            "drift_flag": int(drift.at[week_end, "drift_flag"]) if week_end in drift.index and pd.notna(drift.at[week_end, "drift_flag"]) else 0,
            "state_ok": bool(cov_state.state_ok),
            "is_warm": bool(cov.is_warm(cov_state)),
        }
        if np.all(np.isfinite(x)):
            prev_cov = cov_state.cov_reg.copy()
            cov_state = cov.update(cov_state, x)
            diag = cov_state.last_diagnostics or {}
            if proto is None:
                proto_seed.append(x)
                if len(proto_seed) >= 260:
                    proto = initialize_prototypes_from_history(np.asarray(proto_seed))
            elif cov.is_warm(cov_state):
                result = update_prototypes(
                    proto,
                    x,
                    cov_state.mean,
                    prev_cov,
                    cov_state.cov_reg,
                    len(rows),
                )
                proto = result.state
                probs = _state_probabilities(x, proto, cov_state.cov_reg)
                cluster = int(np.argmax(probs))
                row["state_probs_json"] = json.dumps([round(float(p), 10) for p in probs])
                row["state_label"] = _semantic_label(proto.centroids, cluster)
        else:
            cov_state = cov.update(cov_state, np.array([np.nan, np.nan]))
            diag = cov_state.last_diagnostics or {}

        for key in [
            "maha",
            "huber_weight",
            "eigval_1",
            "eigval_2_raw",
            "eigval_2_reg",
            "condition_number_raw",
            "condition_number_reg",
            "eigval_2_was_floored",
        ]:
            row[key] = diag.get(key, np.nan)
        row["state_ok"] = bool(cov_state.state_ok)
        row["is_warm"] = bool(cov.is_warm(cov_state))
        rows.append(row)

    weekly = pd.DataFrame(rows, columns=REPLAY_COLUMNS)
    events = {
        name: weekly[
            (pd.to_datetime(weekly["week_end"]) >= pd.Timestamp(start))
            & (pd.to_datetime(weekly["week_end"]) <= pd.Timestamp(end))
        ].copy()
        for name, (start, end) in EVENT_WINDOWS.items()
    }
    return ReplayBundle(weekly=weekly, event_windows=events)


def write_replay_outputs(bundle: ReplayBundle, output_dir: str | Path) -> Path:
    """Write deterministic replay CSV outputs."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle.weekly.to_csv(out / "weekly_replay.csv", index=False)
    for name, frame in bundle.event_windows.items():
        frame.to_csv(out / f"event_{name}.csv", index=False)
    return out
