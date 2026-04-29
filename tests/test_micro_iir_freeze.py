from __future__ import annotations

import pytest

from qqq_cycle.core.micro_layer import MicroIIRState, update_weekly_micro_iir_state


def prior_state() -> MicroIIRState:
    return MicroIIRState(
        h_t_lead_prev=0.72,
        heal_count=2,
        envelope_internal_state=0.72,
        breaker_internal_state="healing",
        rho_update_state="strict_observation",
        micro_state_frozen=False,
    )


def test_degraded_backfill_freezes_h_t_lead_prev() -> None:
    before = prior_state()

    after = update_weekly_micro_iir_state(before, h_t_raw=0.1, backfill_mode="degraded_backfill")

    assert after.h_t_lead_prev == before.h_t_lead_prev


def test_degraded_backfill_freezes_heal_count() -> None:
    before = prior_state()

    after = update_weekly_micro_iir_state(before, h_t_raw=0.1, backfill_mode="degraded_backfill")

    assert after.heal_count == before.heal_count


def test_degraded_backfill_prevents_passive_decay() -> None:
    before = prior_state()

    after = update_weekly_micro_iir_state(
        before,
        h_t_raw=0.0,
        backfill_mode="degraded_backfill",
        delta=0.9,
    )

    assert after.h_t_lead_prev == pytest.approx(0.72)
    assert after.h_t_lead_prev != pytest.approx(0.72 * 0.9)


def test_degraded_backfill_emits_micro_state_frozen_true() -> None:
    after = update_weekly_micro_iir_state(
        prior_state(), h_t_raw=None, backfill_mode="degraded_backfill"
    )

    assert after.micro_state_frozen is True


def test_strict_recovery_does_not_freeze_state_by_default() -> None:
    before = prior_state()

    after = update_weekly_micro_iir_state(before, h_t_raw=0.8, backfill_mode="strict_recovery")

    assert after.micro_state_frozen is False
    assert after.h_t_lead_prev == pytest.approx(0.8)
    assert after.heal_count == 0


def test_2026_05_01_consumes_last_real_prior_micro_state() -> None:
    state_before_2026_04_24 = prior_state()

    state_after_2026_04_24_degraded = update_weekly_micro_iir_state(
        state_before_2026_04_24,
        h_t_raw=0.05,
        backfill_mode="degraded_backfill",
    )
    state_after_2026_05_01 = update_weekly_micro_iir_state(
        state_after_2026_04_24_degraded,
        h_t_raw=0.8,
        backfill_mode="strict_recovery",
    )

    assert (
        state_after_2026_04_24_degraded.h_t_lead_prev
        == state_before_2026_04_24.h_t_lead_prev
    )
    assert state_after_2026_04_24_degraded.heal_count == state_before_2026_04_24.heal_count
    assert state_after_2026_04_24_degraded.micro_state_frozen is True
    assert state_after_2026_05_01.h_t_lead_prev == pytest.approx(0.8)
    assert state_after_2026_05_01.rho_update_state == "strict_observation"
