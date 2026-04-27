import math

import pandas as pd
import pytest

from qqq_cycle.core.micro_layer import (
    MicroLayerUnavailableError,
    MicroDailyState,
    compute_breadth,
    compute_correlation_concentration,
    compute_micro_score,
    compute_smoothed_weights,
    iir_envelope_with_breaker,
    matured_member_sets,
    should_hold_for_giant_missing_weight,
    update_micro_daily_state,
    weekly_median_micro,
    z_wrob_156,
)


def test_member_added_on_day_zero_is_not_in_v20_until_day_20() -> None:
    state = MicroDailyState.empty()
    dates = pd.date_range("2024-01-02", periods=20, freq="B")

    for trade_date in dates[:19]:
        state = update_micro_daily_state(state, {"AAPL"}, trade_date)
    v20, v60 = matured_member_sets(state)
    assert "AAPL" not in v20
    assert "AAPL" not in v60

    state = update_micro_daily_state(state, {"AAPL"}, dates[19])
    v20, v60 = matured_member_sets(state)
    assert "AAPL" in v20
    assert "AAPL" not in v60


def test_grace_period_member_is_excluded_and_age_counter_freezes() -> None:
    state = MicroDailyState.empty()
    dates = pd.date_range("2024-01-02", periods=22, freq="B")
    for trade_date in dates[:20]:
        state = update_micro_daily_state(state, {"AAPL", "MSFT"}, trade_date)
    age_before_missing = state.member_ages["MSFT"]

    state = update_micro_daily_state(state, {"AAPL"}, dates[20], grace_period_days=3)
    v20, _ = matured_member_sets(state)

    assert "MSFT" in state.grace_members
    assert state.member_ages["MSFT"] == age_before_missing
    assert "MSFT" not in v20
    assert state.data_contaminated is True

    state = update_micro_daily_state(state, {"AAPL", "MSFT"}, dates[21], grace_period_days=3)
    assert "MSFT" not in state.grace_members
    assert state.member_ages["MSFT"] == age_before_missing + 1


def test_giant_missing_weight_check_holds_when_grace_weight_exceeds_half_fifth_weight() -> None:
    state = update_micro_daily_state(
        MicroDailyState.empty(),
        {"A", "B", "C", "D", "E", "F"},
        pd.Timestamp("2024-01-02"),
    ).with_smoothed_weights(
        {
            "A": 0.30,
            "B": 0.20,
            "C": 0.15,
            "D": 0.10,
            "E": 0.08,
            "F": 0.07,
        }
    )
    state = update_micro_daily_state(state, {"B", "C", "D", "E", "F"}, pd.Timestamp("2024-01-03"))

    decision = should_hold_for_giant_missing_weight(state)

    assert decision.hold_micro_recompute is True
    assert decision.data_contaminated is True
    assert decision.missing_weight == 0.30
    assert decision.threshold == 0.04


def test_smoothed_weights_use_five_day_half_life_and_freeze_on_rule_days() -> None:
    previous = {"A": 0.6, "B": 0.4}
    lagged = {"A": 0.2, "B": 0.8}

    frozen = compute_smoothed_weights(previous, lagged, is_rule_window=True)
    updated = compute_smoothed_weights(previous, lagged, is_rule_window=False)

    rho = 2 ** (-1 / 5)
    assert frozen == previous
    assert updated["A"] == pytest.approx(rho * 0.6 + (1 - rho) * 0.2)
    assert updated["B"] == pytest.approx(rho * 0.4 + (1 - rho) * 0.8)


class _FixturePITEngine:
    def __init__(self, windows: dict[str, pd.Series]) -> None:
        self.windows = windows
        self.calls: list[tuple[str, pd.Timestamp, int, pd.Timestamp]] = []

    def get_adjusted_window(
        self, ticker: str, end_date: pd.Timestamp, window: int, asof: pd.Timestamp
    ) -> pd.Series:
        self.calls.append((ticker, pd.Timestamp(end_date), window, pd.Timestamp(asof)))
        return self.windows[ticker].tail(window)


def test_compute_breadth_is_zero_when_all_members_above_pit_ma20() -> None:
    idx = pd.date_range("2024-01-02", periods=20, freq="B")
    windows = {
        "A": pd.Series([100.0] * 19 + [110.0], index=idx),
        "B": pd.Series([50.0] * 19 + [55.0], index=idx),
    }
    engine = _FixturePITEngine(windows)

    breadth = compute_breadth(
        members=frozenset({"A", "B"}),
        smoothed_weights={"A": 0.75, "B": 0.25},
        trade_date=idx[-1],
        pit_engine=engine,
    )

    assert breadth == pytest.approx(0.0)
    assert engine.calls == [
        ("A", idx[-1], 20, idx[-1]),
        ("B", idx[-1], 20, idx[-1]),
    ]


def test_compute_breadth_raises_when_pit_engine_unavailable() -> None:
    with pytest.raises(MicroLayerUnavailableError):
        compute_breadth(
            members=frozenset({"A"}),
            smoothed_weights={"A": 1.0},
            trade_date=pd.Timestamp("2024-01-31"),
            pit_engine=None,
        )


def test_compute_correlation_concentration_is_one_for_perfect_correlation() -> None:
    idx = pd.date_range("2024-01-02", periods=60, freq="B")
    base = pd.Series(range(100, 160), index=idx, dtype=float)
    price_windows = {"A": base, "B": base * 2.0, "C": base * 3.0}

    concentration = compute_correlation_concentration(
        members=frozenset(price_windows),
        smoothed_weights={"A": 0.5, "B": 0.3, "C": 0.2},
        price_windows=price_windows,
    )

    assert concentration == pytest.approx(1.0)


def test_weekly_median_micro_uses_only_days_inside_each_week() -> None:
    daily = pd.DataFrame(
        {
            "b_tau": [0.1, 0.5, 0.9, 0.2],
            "c_tau": [0.2, 0.6, 1.0, 0.3],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"]),
    )

    weekly = weekly_median_micro(daily)

    assert list(weekly.index) == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")]
    assert weekly.loc[pd.Timestamp("2024-01-05"), "b_wk"] == pytest.approx(0.5)
    assert weekly.loc[pd.Timestamp("2024-01-12"), "b_wk"] == pytest.approx(0.2)


def test_z_wrob_156_downweights_rule_week_history() -> None:
    idx = pd.date_range("2024-01-05", periods=157, freq="W-FRI")
    series = pd.Series([0.0] * 76 + [10.0] * 80 + [0.0], index=idx)
    weights = pd.Series([1.0] * 76 + [0.3] * 80 + [1.0], index=idx)

    z = z_wrob_156(series, weights=weights, eps=1e-6)
    z_unweighted = z_wrob_156(series, weights=pd.Series(1.0, index=idx), eps=1e-6)

    assert z.iloc[-1] > z_unweighted.iloc[-1]


def test_compute_micro_score_maps_raw_average_through_logistic() -> None:
    score = compute_micro_score(b_tilde=2.0, c_tilde=0.0)

    assert score.raw == pytest.approx(1.0)
    assert score.h_t == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))


def test_iir_two_week_recovery_does_not_clear_memory() -> None:
    x1 = iir_envelope_with_breaker(0.9, 0.9, 0.9, x_lead_prev=0.0)
    x2 = iir_envelope_with_breaker(0.1, 0.9, 0.9, x_lead_prev=x1)
    x3 = iir_envelope_with_breaker(0.1, 0.1, 0.9, x_lead_prev=x2)

    assert x1 == pytest.approx(0.4)
    assert x2 == pytest.approx(0.36)
    assert x3 == pytest.approx(0.324)


def test_iir_three_week_recovery_clears_memory() -> None:
    x1 = iir_envelope_with_breaker(0.9, 0.9, 0.9, x_lead_prev=0.0)
    x2 = iir_envelope_with_breaker(0.1, 0.9, 0.9, x_lead_prev=x1)
    x3 = iir_envelope_with_breaker(0.1, 0.1, 0.9, x_lead_prev=x2)
    x4 = iir_envelope_with_breaker(0.1, 0.1, 0.1, x_lead_prev=x3)

    assert x4 == 0.0


def test_iir_delta_decay_over_ten_weeks_with_no_new_signal() -> None:
    x = iir_envelope_with_breaker(0.9, 0.9, 0.9, x_lead_prev=0.0)
    for _ in range(10):
        x = iir_envelope_with_breaker(0.5, 0.5, 0.5, x_lead_prev=x, delta=0.9)

    assert x == pytest.approx(0.4 * 0.9**10)
