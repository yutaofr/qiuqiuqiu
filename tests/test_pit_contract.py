import pandas as pd
import pytest
import numpy as np

from qqq_cycle.data_contracts.pit_adjustment import (
    DataNotAvailableError,
    HindsightAdjustedDataError,
    InMemoryPITAdjustmentEngine,
    InsufficientHistoryError,
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


def test_in_memory_engine_adjusts_window_to_asof_basis() -> None:
    bars = [
        PITPriceBar(
            trade_date=pd.Timestamp("2024-01-02"),
            ticker="QQQ",
            raw_close=100.0,
            split_factor_cum_pti=1.0,
            dividend_factor_cum_pti=1.0,
            adj_close_pti=100.0,
            asof_timestamp=pd.Timestamp("2024-01-02 16:00"),
        ),
        PITPriceBar(
            trade_date=pd.Timestamp("2024-01-03"),
            ticker="QQQ",
            raw_close=51.0,
            split_factor_cum_pti=2.0,
            dividend_factor_cum_pti=1.0,
            adj_close_pti=102.0,
            asof_timestamp=pd.Timestamp("2024-01-03 16:00"),
        ),
    ]
    engine = InMemoryPITAdjustmentEngine(bars)

    window = engine.get_adjusted_window(
        "QQQ",
        end_date=pd.Timestamp("2024-01-03"),
        window=2,
        asof=pd.Timestamp("2024-01-03 16:00"),
    )

    np.testing.assert_allclose(window.to_numpy(), [200.0, 51.0])


def test_in_memory_engine_rejects_future_asof_and_insufficient_history() -> None:
    bars = [
        PITPriceBar(
            trade_date=pd.Timestamp("2024-01-02"),
            ticker="QQQ",
            raw_close=100.0,
            split_factor_cum_pti=1.0,
            dividend_factor_cum_pti=1.0,
            adj_close_pti=100.0,
            asof_timestamp=pd.Timestamp("2024-01-02 16:00"),
        )
    ]
    engine = InMemoryPITAdjustmentEngine(bars)

    with pytest.raises(DataNotAvailableError):
        engine.get_adj_close(
            "QQQ",
            trade_date=pd.Timestamp("2024-01-02"),
            asof=pd.Timestamp("2024-01-02 15:59"),
        )
    with pytest.raises(InsufficientHistoryError):
        engine.get_adjusted_window(
            "QQQ",
            end_date=pd.Timestamp("2024-01-02"),
            window=2,
            asof=pd.Timestamp("2024-01-02 16:00"),
        )
