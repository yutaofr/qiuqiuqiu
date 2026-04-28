"""Live state persistence: save/load LiveState to a directory of JSON + .npy files.

Layout of a state directory (e.g. state/live_state_latest/):
    manifest.json                    — all scalar fields + array filenames
    macro_tail.csv                   — historical macro tail for rolling batch layers
    cov_mean.npy, cov_raw.npy, ...   — CovarianceState2D arrays
    proto_centroids.npy, ...         — PrototypeState arrays (absent when proto is None)
    proto_reactivation_buffers.json  — PrototypeState buffer lists
    proto_seed.npy                   — warmup seed rows (absent when proto is initialized)
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qqq_cycle.core.covariance import CovarianceState2D
from qqq_cycle.core.proto_online import PrototypeState, ReactivationBuffer

# Keep at most this many rows in the macro tail to bound disk/memory usage.
# 600 rows (≈11.5 years weekly) comfortably covers the 520-week rolling windows.
_MAX_TAIL_ROWS = 600


class StateNotAvailableError(Exception):
    """Raised when state cannot be loaded (missing, corrupt, or incomplete)."""


@dataclass
class LiveState:
    """Complete live-engine state for one weekly snapshot.

    Encapsulates all mutable state that must survive between weekly runs:
    pipeline state (covariance, prototype, IIR envelope), portfolio state
    (circuit breaker), and the macro tail needed for batch layer computations.
    """

    week_end: str
    # Pipeline state
    cov_state: CovarianceState2D
    proto: PrototypeState | None
    proto_seed: list[np.ndarray]
    h_t_lead_prev: float
    heal_count: int
    warmup_count: int
    # Portfolio state
    breaker_active: bool
    weeks_outside_s1: int
    prev_omega_qqq: float
    # Rolling macro tail for batch-layer computations (state_layer, stress_layer, drift)
    macro_tail: pd.DataFrame
    # Bookkeeping
    last_successful_timestamps: dict[str, str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_array(arr: np.ndarray, path: Path) -> None:
    np.save(path, arr)


def _load_array(path: Path, name: str) -> np.ndarray:
    if not path.exists():
        raise StateNotAvailableError(f"state array file missing: {name} at {path}")
    return np.load(path)


def _serialize_cov_state(state: CovarianceState2D, arrays_dir: Path) -> dict[str, Any]:
    _save_array(state.mean, arrays_dir / "cov_mean.npy")
    _save_array(state.cov_raw, arrays_dir / "cov_raw.npy")
    _save_array(state.cov_reg, arrays_dir / "cov_reg.npy")
    _save_array(state.eigvals, arrays_dir / "cov_eigvals.npy")
    _save_array(state.eigvecs, arrays_dir / "cov_eigvecs.npy")
    return {
        "warmup_count": int(state.warmup_count),
        "pending_missing_steps": int(state.pending_missing_steps),
        "state_ok": bool(state.state_ok),
    }


def _deserialize_cov_state(d: dict[str, Any], arrays_dir: Path) -> CovarianceState2D:
    return CovarianceState2D(
        mean=_load_array(arrays_dir / "cov_mean.npy", "cov_mean"),
        cov_raw=_load_array(arrays_dir / "cov_raw.npy", "cov_raw"),
        cov_reg=_load_array(arrays_dir / "cov_reg.npy", "cov_reg"),
        eigvals=_load_array(arrays_dir / "cov_eigvals.npy", "cov_eigvals"),
        eigvecs=_load_array(arrays_dir / "cov_eigvecs.npy", "cov_eigvecs"),
        warmup_count=int(d["warmup_count"]),
        pending_missing_steps=int(d["pending_missing_steps"]),
        last_diagnostics=None,
        state_ok=bool(d["state_ok"]),
    )


def _serialize_proto_state(
    proto: PrototypeState, arrays_dir: Path
) -> dict[str, Any]:
    _save_array(proto.centroids, arrays_dir / "proto_centroids.npy")
    _save_array(proto.weights, arrays_dir / "proto_weights.npy")
    _save_array(proto.residuals, arrays_dir / "proto_residuals.npy")
    _save_array(proto.last_active, arrays_dir / "proto_last_active.npy")
    buffers_data = [
        {"samples": [s.tolist() for s in buf.samples], "active": buf.active}
        for buf in proto.reactivation_buffers
    ]
    (arrays_dir / "proto_reactivation_buffers.json").write_text(
        json.dumps(buffers_data), encoding="utf-8"
    )
    return {}


def _deserialize_proto_state(arrays_dir: Path) -> PrototypeState:
    centroids = _load_array(arrays_dir / "proto_centroids.npy", "proto_centroids")
    weights = _load_array(arrays_dir / "proto_weights.npy", "proto_weights")
    residuals = _load_array(arrays_dir / "proto_residuals.npy", "proto_residuals")
    last_active = _load_array(arrays_dir / "proto_last_active.npy", "proto_last_active")
    buf_path = arrays_dir / "proto_reactivation_buffers.json"
    if not buf_path.exists():
        raise StateNotAvailableError(f"proto_reactivation_buffers.json missing at {arrays_dir}")
    raw_buffers = json.loads(buf_path.read_text(encoding="utf-8"))
    buffers = [
        ReactivationBuffer(
            samples=[np.array(s, dtype=float) for s in b["samples"]],
            active=bool(b["active"]),
        )
        for b in raw_buffers
    ]
    return PrototypeState(
        centroids=centroids,
        weights=weights,
        residuals=residuals,
        last_active=last_active,
        reactivation_buffers=buffers,
    )


def _serialize_proto_seed(seed: list[np.ndarray], arrays_dir: Path) -> int:
    if not seed:
        return 0
    arr = np.vstack(seed)
    _save_array(arr, arrays_dir / "proto_seed.npy")
    return len(seed)


def _deserialize_proto_seed(arrays_dir: Path, count: int) -> list[np.ndarray]:
    if count == 0:
        return []
    path = arrays_dir / "proto_seed.npy"
    if not path.exists():
        raise StateNotAvailableError(f"proto_seed.npy missing at {arrays_dir}")
    arr = np.load(path)
    return [arr[i] for i in range(len(arr))]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_state(state: LiveState, state_dir: Path) -> None:
    """Persist LiveState to state_dir/live_state_latest/ and a dated archive.

    Creates two full copies: latest (always overwritten) and a dated archive
    named after state.week_end. Never symlinks — both are independent copies.
    """
    latest_dir = state_dir / "live_state_latest"
    dated_dir = state_dir / f"live_state_{state.week_end.replace('-', '')}"

    _write_state_to_dir(state, latest_dir)
    if dated_dir != latest_dir:
        if dated_dir.exists():
            shutil.rmtree(dated_dir)
        shutil.copytree(latest_dir, dated_dir)


def _write_state_to_dir(state: LiveState, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    cov_scalar = _serialize_cov_state(state.cov_state, target)

    proto_present = state.proto is not None
    if proto_present:
        _serialize_proto_state(state.proto, target)  # type: ignore[arg-type]

    proto_seed_count = _serialize_proto_seed(state.proto_seed, target)

    # Trim macro tail before saving
    tail = state.macro_tail.tail(_MAX_TAIL_ROWS)
    tail.to_csv(target / "macro_tail.csv")

    manifest: dict[str, Any] = {
        "week_end": state.week_end,
        "h_t_lead_prev": float(state.h_t_lead_prev),
        "heal_count": int(state.heal_count),
        "warmup_count": int(state.warmup_count),
        "breaker_active": bool(state.breaker_active),
        "weeks_outside_s1": int(state.weeks_outside_s1),
        "prev_omega_qqq": float(state.prev_omega_qqq),
        "proto_present": proto_present,
        "proto_seed_count": proto_seed_count,
        "last_successful_timestamps": state.last_successful_timestamps,
        "cov": cov_scalar,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_state(state_dir: Path) -> LiveState:
    """Load LiveState from state_dir/live_state_latest/.

    Raises StateNotAvailableError if the directory is missing, the manifest is
    absent or corrupt, or any required array file is missing. Never silently
    falls back to a full history recompute.
    """
    latest_dir = state_dir / "live_state_latest"
    manifest_path = latest_dir / "manifest.json"
    if not latest_dir.exists() or not manifest_path.exists():
        raise StateNotAvailableError(
            f"live state not found at {latest_dir}. "
            "Run scripts/bootstrap_live_state.py to initialize."
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise StateNotAvailableError(f"manifest.json corrupt: {exc}") from exc

    try:
        cov_state = _deserialize_cov_state(manifest["cov"], latest_dir)

        proto: PrototypeState | None = None
        if manifest["proto_present"]:
            proto = _deserialize_proto_state(latest_dir)

        proto_seed = _deserialize_proto_seed(latest_dir, manifest["proto_seed_count"])

        tail_path = latest_dir / "macro_tail.csv"
        if not tail_path.exists():
            raise StateNotAvailableError(f"macro_tail.csv missing at {latest_dir}")
        macro_tail = pd.read_csv(tail_path, index_col=0, parse_dates=True)

        return LiveState(
            week_end=str(manifest["week_end"]),
            cov_state=cov_state,
            proto=proto,
            proto_seed=proto_seed,
            h_t_lead_prev=float(manifest["h_t_lead_prev"]),
            heal_count=int(manifest["heal_count"]),
            warmup_count=int(manifest["warmup_count"]),
            breaker_active=bool(manifest["breaker_active"]),
            weeks_outside_s1=int(manifest["weeks_outside_s1"]),
            prev_omega_qqq=float(manifest["prev_omega_qqq"]),
            macro_tail=macro_tail,
            last_successful_timestamps=dict(manifest["last_successful_timestamps"]),
        )
    except StateNotAvailableError:
        raise
    except (KeyError, ValueError, OSError) as exc:
        raise StateNotAvailableError(f"state load failed: {exc}") from exc
