"""Tests for live state serialization and deserialization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qqq_cycle.core.covariance import CovarianceState2D, RobustEWCov2D
from qqq_cycle.core.proto_online import PrototypeState, ReactivationBuffer, initialize_prototypes_from_history
from qqq_cycle.live.state_io import LiveState, StateNotAvailableError, load_state, save_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cov_state() -> CovarianceState2D:
    rng = np.random.default_rng(42)
    hist = rng.standard_normal((25, 2))
    cov = RobustEWCov2D()
    state = cov.initialize_from_history(hist)
    for row in hist[20:]:
        state = cov.update(state, row)
    return state


def _make_proto_state() -> PrototypeState:
    rng = np.random.default_rng(42)
    hist = rng.standard_normal((50, 2))
    return initialize_prototypes_from_history(hist)


def _make_macro_tail() -> pd.DataFrame:
    dates = pd.date_range("2020-01-03", periods=10, freq="W-FRI")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.standard_normal((10, 8)),
        index=dates,
        columns=["DFII10", "DGS2", "BAMLH0A0HYM2", "NFCI", "VIXCLS", "AI_GPR", "USEPUINDXD", "QQQ"],
    )


def _make_live_state(with_proto: bool = True) -> LiveState:
    cov_state = _make_cov_state()
    proto = _make_proto_state() if with_proto else None
    return LiveState(
        week_end="2024-01-05",
        cov_state=cov_state,
        proto=proto,
        proto_seed=[],
        h_t_lead_prev=0.42,
        heal_count=1,
        warmup_count=cov_state.warmup_count,
        breaker_active=False,
        weeks_outside_s1=2,
        prev_omega_qqq=0.65,
        macro_tail=_make_macro_tail(),
        last_successful_timestamps={"last_run": "2024-01-05T00:00:00+00:00"},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip_with_proto(tmp_path: Path) -> None:
    """Save a LiveState with proto and reload it; all fields should be equal."""
    state = _make_live_state(with_proto=True)
    save_state(state, tmp_path)
    restored = load_state(tmp_path)

    assert restored.week_end == state.week_end
    assert restored.h_t_lead_prev == pytest.approx(state.h_t_lead_prev)
    assert restored.heal_count == state.heal_count
    assert restored.warmup_count == state.warmup_count
    assert restored.breaker_active == state.breaker_active
    assert restored.weeks_outside_s1 == state.weeks_outside_s1
    assert restored.prev_omega_qqq == pytest.approx(state.prev_omega_qqq)

    np.testing.assert_array_almost_equal(restored.cov_state.mean, state.cov_state.mean)
    np.testing.assert_array_almost_equal(restored.cov_state.cov_raw, state.cov_state.cov_raw)
    np.testing.assert_array_almost_equal(restored.cov_state.cov_reg, state.cov_state.cov_reg)
    assert restored.cov_state.warmup_count == state.cov_state.warmup_count

    assert restored.proto is not None
    np.testing.assert_array_almost_equal(restored.proto.centroids, state.proto.centroids)
    np.testing.assert_array_almost_equal(restored.proto.weights, state.proto.weights)
    np.testing.assert_array_almost_equal(restored.proto.residuals, state.proto.residuals)
    np.testing.assert_array_almost_equal(restored.proto.last_active, state.proto.last_active)
    assert len(restored.proto.reactivation_buffers) == len(state.proto.reactivation_buffers)

    pd.testing.assert_frame_equal(restored.macro_tail, state.macro_tail, check_freq=False)


def test_save_and_load_roundtrip_without_proto(tmp_path: Path) -> None:
    """Save a LiveState without proto (warmup phase) and reload it."""
    rng = np.random.default_rng(7)
    seed_rows = [rng.standard_normal(2) for _ in range(10)]
    state = LiveState(
        week_end="2023-06-02",
        cov_state=_make_cov_state(),
        proto=None,
        proto_seed=seed_rows,
        h_t_lead_prev=0.0,
        heal_count=0,
        warmup_count=10,
        breaker_active=False,
        weeks_outside_s1=0,
        prev_omega_qqq=0.5,
        macro_tail=_make_macro_tail(),
        last_successful_timestamps={},
    )
    save_state(state, tmp_path)
    restored = load_state(tmp_path)

    assert restored.proto is None
    assert len(restored.proto_seed) == len(state.proto_seed)
    for orig, loaded in zip(state.proto_seed, restored.proto_seed):
        np.testing.assert_array_almost_equal(orig, loaded)


def test_load_missing_state_raises(tmp_path: Path) -> None:
    """load_state raises StateNotAvailableError when latest directory is absent."""
    with pytest.raises(StateNotAvailableError, match="live state not found"):
        load_state(tmp_path)


def test_load_missing_manifest_raises(tmp_path: Path) -> None:
    """StateNotAvailableError when manifest.json is absent."""
    (tmp_path / "live_state_latest").mkdir()
    with pytest.raises(StateNotAvailableError):
        load_state(tmp_path)


def test_corrupt_manifest_raises(tmp_path: Path) -> None:
    """StateNotAvailableError when manifest.json is corrupt JSON."""
    latest = tmp_path / "live_state_latest"
    latest.mkdir()
    (latest / "manifest.json").write_text("not valid json{{{", encoding="utf-8")
    with pytest.raises(StateNotAvailableError, match="corrupt"):
        load_state(tmp_path)


def test_missing_npy_raises(tmp_path: Path) -> None:
    """StateNotAvailableError when manifest is valid but an array file is missing."""
    state = _make_live_state(with_proto=False)
    save_state(state, tmp_path)
    (tmp_path / "live_state_latest" / "cov_mean.npy").unlink()
    with pytest.raises(StateNotAvailableError):
        load_state(tmp_path)


def test_json_sidecar_written(tmp_path: Path) -> None:
    """manifest.json is present and contains the expected fields after save."""
    state = _make_live_state(with_proto=True)
    save_state(state, tmp_path)
    manifest = json.loads((tmp_path / "live_state_latest" / "manifest.json").read_text())
    assert manifest["week_end"] == state.week_end
    assert manifest["warmup_count"] == state.warmup_count
    assert manifest["breaker_active"] == state.breaker_active
    assert "cov" in manifest
    assert "last_successful_timestamps" in manifest


def test_dated_archive_created(tmp_path: Path) -> None:
    """save_state creates both live_state_latest/ and a dated archive directory."""
    state = _make_live_state(with_proto=False)
    save_state(state, tmp_path)
    dated = tmp_path / "live_state_20240105"
    assert (tmp_path / "live_state_latest").exists()
    assert dated.exists()
    # Both are independent copies — modifying one doesn't affect the other.
    (tmp_path / "live_state_latest" / "manifest.json").write_text("{}", encoding="utf-8")
    dated_manifest = json.loads((dated / "manifest.json").read_text())
    assert dated_manifest["week_end"] == state.week_end


def test_macro_tail_trimmed_to_max(tmp_path: Path) -> None:
    """macro_tail is trimmed to _MAX_TAIL_ROWS on save."""
    from qqq_cycle.live.state_io import _MAX_TAIL_ROWS

    dates = pd.date_range("2000-01-07", periods=_MAX_TAIL_ROWS + 50, freq="W-FRI")
    rng = np.random.default_rng(1)
    big_tail = pd.DataFrame(
        rng.standard_normal((len(dates), 8)),
        index=dates,
        columns=["DFII10", "DGS2", "BAMLH0A0HYM2", "NFCI", "VIXCLS", "AI_GPR", "USEPUINDXD", "QQQ"],
    )
    state = LiveState(
        week_end=dates[-1].strftime("%Y-%m-%d"),
        cov_state=_make_cov_state(),
        proto=None,
        proto_seed=[],
        h_t_lead_prev=0.0,
        heal_count=0,
        warmup_count=5,
        breaker_active=False,
        weeks_outside_s1=0,
        prev_omega_qqq=0.5,
        macro_tail=big_tail,
        last_successful_timestamps={},
    )
    save_state(state, tmp_path)
    restored = load_state(tmp_path)
    assert len(restored.macro_tail) == _MAX_TAIL_ROWS
