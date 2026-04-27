import pytest

from qqq_cycle.core.interpretability import (
    Attribution,
    ContaminationFlags,
    DriftDiagnostics,
    InterpretabilityRecord,
    ModuleHealth,
    build_interpretability,
)


def test_build_interpretability_round_trips_all_fields_without_loss() -> None:
    record = build_interpretability(
        L_t=1.0,
        T_t=2.0,
        P_t=3.0,
        delta4_L_t=4.0,
        delta4_T_t=5.0,
        delta4_P_t=6.0,
        g_tilde=0.7,
        e_tilde=0.8,
        b_tilde=0.9,
        c_tilde=1.0,
        omega_t=0.6,
        s_t=0.5,
        n_t=0.4,
        eta_t=0.3,
        is_rule_week=True,
        has_constituent_change=False,
        data_contaminated=True,
        v60_count=79,
        universe_count=100,
        delta_abs_raw=1.2,
        d_state=1.1,
        g_stress=-1.2,
        micro_raw=1.3,
        module_health=ModuleHealth(
            h_macro=1,
            h_exo=1,
            h_micro=0,
            h_state=1,
        ),
    )

    assert isinstance(record, InterpretabilityRecord)
    assert isinstance(record.A_t, Attribution)
    assert record.A_t.H_components == pytest.approx((0.40, 0.70, 0.75))
    assert record.A_t.I_components == pytest.approx((2.0, 1.5, 1.2))
    assert record.A_t.stress_components == pytest.approx((0.7, 0.8))
    assert record.A_t.micro_components == pytest.approx((0.9, 1.0))
    assert record.A_t.rho_components == pytest.approx((0.3, 0.3, 0.3))
    assert record.C_t == ContaminationFlags(
        c_rule=1,
        c_const=0,
        c_data=1,
        c_micro=1,
        c_drift=0,
    )
    assert record.D_t == DriftDiagnostics(
        d_state=1.1,
        d_stress=1.2,
        d_frag=1.3,
        d_abs=1.2,
    )
    assert record.H_t == ModuleHealth(h_macro=1, h_exo=1, h_micro=0, h_state=1)


def test_drift_contamination_flag_triggers_at_hi_threshold() -> None:
    record = build_interpretability(
        L_t=0.0,
        T_t=0.0,
        P_t=0.0,
        delta4_L_t=0.0,
        delta4_T_t=0.0,
        delta4_P_t=0.0,
        g_tilde=0.0,
        e_tilde=0.0,
        b_tilde=0.0,
        c_tilde=0.0,
        omega_t=0.0,
        s_t=0.0,
        n_t=0.0,
        eta_t=0.0,
        is_rule_week=False,
        has_constituent_change=False,
        data_contaminated=False,
        v60_count=100,
        universe_count=100,
        delta_abs_raw=-1.8,
        d_state=0.0,
        g_stress=0.0,
        micro_raw=0.0,
        module_health=ModuleHealth(1, 1, 1, 1),
    )

    assert record.C_t.c_drift == 1
