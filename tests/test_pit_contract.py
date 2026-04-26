import pandas as pd
import pytest

from qqq_cycle.data_contracts.pit_adjustment import (
    DataNotAvailableError,
    HindsightAdjustedDataError,
    PITAdjustmentEngine,
    PITPriceBar,
    degrade_micro_mode,
)


def test_pit_price_bar_validates_adjusted_identity() -> None:
    bar = PITPriceBar(
        trade_date=pd.Timestamp("2024-01-02"),
        ticker="QQQ",
        raw_close=100.0,
        split_factor_cum_pti=2.0,
        dividend_factor_cum_pti=0.5,
        adj_close_pti=100.0,
        asof_timestamp=pd.Timestamp("2024-01-02 16:00"),
    )

    assert bar.adj_close_pti == 100.0


def test_pit_price_bar_rejects_future_asof_and_identity_mismatch() -> None:
    with pytest.raises(ValueError):
        PITPriceBar(
            trade_date=pd.Timestamp("2024-01-02"),
            ticker="QQQ",
            raw_close=100.0,
            split_factor_cum_pti=1.0,
            dividend_factor_cum_pti=1.0,
            adj_close_pti=101.0,
            asof_timestamp=pd.Timestamp("2024-01-01"),
        )


def test_base_engine_fails_closed_when_pit_data_unavailable() -> None:
    engine = PITAdjustmentEngine()

    with pytest.raises(DataNotAvailableError):
        engine.get_adj_close("QQQ", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02"))
    with pytest.raises(DataNotAvailableError):
        engine.get_adjusted_window("QQQ", pd.Timestamp("2024-01-02"), 20, pd.Timestamp("2024-01-02"))

    degraded = degrade_micro_mode(engine)
    assert degraded.micro_enabled is False
    assert degraded.h_t is None
    assert degraded.rho_t is None


def test_hindsight_adjusted_source_is_forbidden() -> None:
    engine = PITAdjustmentEngine(only_hindsight_adjusted_available=True)

    with pytest.raises(HindsightAdjustedDataError):
        engine.get_adj_close("QQQ", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02"))
