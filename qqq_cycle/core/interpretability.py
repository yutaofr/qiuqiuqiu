"""Pure interpretability record assembly for the QQQ cycle-state system."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Attribution:
    """Layer attribution components from model spec §11.1."""

    H_components: tuple[float, float, float]
    I_components: tuple[float, float, float]
    stress_components: tuple[float, float]
    micro_components: tuple[float, float]
    rho_components: tuple[float, float, float]


@dataclass(frozen=True)
class ContaminationFlags:
    """Point-in-time contamination flags from model spec §11.2."""

    c_rule: int
    c_const: int
    c_data: int
    c_micro: int
    c_drift: int


@dataclass(frozen=True)
class DriftDiagnostics:
    """Drift and fragility diagnostics from model spec §11.3."""

    d_state: float
    d_stress: float
    d_frag: float
    d_abs: float


@dataclass(frozen=True)
class ModuleHealth:
    """Module health vector from model spec §11.4."""

    h_macro: int
    h_exo: int
    h_micro: int
    h_state: int


@dataclass(frozen=True)
class InterpretabilityRecord:
    """Auditable I_t = (A_t, C_t, D_t, H_t) output tuple."""

    A_t: Attribution
    C_t: ContaminationFlags
    D_t: DriftDiagnostics
    H_t: ModuleHealth


def build_interpretability(
    *,
    L_t: float,
    T_t: float,
    P_t: float,
    delta4_L_t: float,
    delta4_T_t: float,
    delta4_P_t: float,
    g_tilde: float,
    e_tilde: float,
    b_tilde: float,
    c_tilde: float,
    omega_t: float,
    s_t: float,
    n_t: float,
    eta_t: float,
    is_rule_week: bool,
    has_constituent_change: bool,
    data_contaminated: bool,
    v60_count: int,
    universe_count: int,
    delta_abs_raw: float,
    d_state: float,
    g_stress: float,
    micro_raw: float,
    module_health: ModuleHealth,
    lambda_rho: float = 0.75,
    theta_drift_hi: float = 1.8,
) -> InterpretabilityRecord:
    """Assemble an interpretability record without I/O.

    Inputs are already-computed weekly layer outputs available at the decision
    timestamp. The function does not read data, recompute state probabilities,
    or perform any adjusted-close lookup.
    """

    if universe_count < 0 or v60_count < 0:
        raise ValueError("member counts must be non-negative")
    mature_ratio = 0.0 if universe_count == 0 else float(v60_count) / float(universe_count)
    attribution = Attribution(
        H_components=(0.40 * float(L_t), 0.35 * float(T_t), 0.25 * float(P_t)),
        I_components=(
            0.50 * float(delta4_L_t),
            0.30 * float(delta4_T_t),
            0.20 * float(delta4_P_t),
        ),
        stress_components=(float(g_tilde), float(e_tilde)),
        micro_components=(float(b_tilde), float(c_tilde)),
        rho_components=(float(omega_t) * float(s_t), float(lambda_rho) * float(n_t), float(eta_t)),
    )
    contamination = ContaminationFlags(
        c_rule=int(bool(is_rule_week)),
        c_const=int(bool(has_constituent_change)),
        c_data=int(bool(data_contaminated)),
        c_micro=int(mature_ratio < 0.80),
        c_drift=int(abs(float(delta_abs_raw)) >= float(theta_drift_hi)),
    )
    diagnostics = DriftDiagnostics(
        d_state=float(d_state),
        d_stress=abs(float(g_stress)),
        d_frag=abs(float(micro_raw)),
        d_abs=abs(float(delta_abs_raw)),
    )
    return InterpretabilityRecord(
        A_t=attribution,
        C_t=contamination,
        D_t=diagnostics,
        H_t=module_health,
    )
