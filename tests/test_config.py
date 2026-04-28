from qqq_cycle.config import load_config


def test_load_config_returns_typed_model_v22_defaults() -> None:
    config = load_config()

    assert config.warmup_weeks == 260
    assert config.dual_memory.robust_window_weeks == 104
    assert config.dual_memory.ew_half_life_weeks == 260
    assert config.covariance.half_life_weeks == 78
    assert config.drift.theta_lo == 1.2
    assert config.drift.theta_hi == 1.8
    assert config.micro.iir_delta == 0.9
    assert config.micro.heal_threshold == 0.25
    assert config.risk.lambda_rho == 0.75
    assert config.risk.omega_state == (1.0, 0.7, 0.3, 0.6, 0.9)
    assert config.ops.operational_timezone == "America/New_York"
    assert config.ops.sla_cutoff_weekday == "SAT"
    assert config.ops.sla_cutoff_time == "12:00"
    assert config.percentile_window_weeks == 520
    assert config.noise_quantile == 0.10
