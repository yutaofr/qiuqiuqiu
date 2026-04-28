import pandas as pd
import pytest
import numpy as np

from qqq_cycle.data_contracts.pit_adjustment import (
    DataNotAvailableError,
    HindsightAdjustedDataError,
    InMemoryPITAdjustmentEngine,
    InsufficientHistoryError,
    LedgerPITAdjustmentEngine,
    PITAdjustmentEngine,
    PITPriceBar,
    degrade_micro_mode,
)
from qqq_cycle.data_contracts.raw_prices import RawPriceObservation, InMemoryRawPriceStore
from qqq_cycle.data_contracts.corp_actions import (
    CorporateActionEvent,
    InMemoryCorporateActionStore,
)
from qqq_cycle.data_contracts.symbol_identity import (
    InMemorySymbolIdentityResolver,
    SymbolIdentityRecord,
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

    assert engine.source_label == "abstract"
    assert engine.asof_semantics == "strict_pit"
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


def test_pit_chained_corporate_action_precision() -> None:
    bars = [
        PITPriceBar(
            trade_date=pd.Timestamp("2021-08-05"),
            ticker="AAPL",
            raw_close=120.0,
            split_factor_cum_pti=1.0,
            dividend_factor_cum_pti=1.0,
            adj_close_pti=120.0,
            asof_timestamp=pd.Timestamp("2021-08-05 16:00"),
        ),
        PITPriceBar(
            trade_date=pd.Timestamp("2021-08-06"),
            ticker="AAPL",
            raw_close=60.0,
            split_factor_cum_pti=2.0,
            dividend_factor_cum_pti=1.0,
            adj_close_pti=120.0,
            asof_timestamp=pd.Timestamp("2021-08-06 16:00"),
        ),
        PITPriceBar(
            trade_date=pd.Timestamp("2021-09-10"),
            ticker="AAPL",
            raw_close=30.0,
            split_factor_cum_pti=8.0,
            dividend_factor_cum_pti=1.0,
            adj_close_pti=240.0,
            asof_timestamp=pd.Timestamp("2021-09-10 16:00"),
        ),
        PITPriceBar(
            trade_date=pd.Timestamp("2021-08-05"),
            ticker="AAPL",
            raw_close=120.0,
            split_factor_cum_pti=1.0,
            dividend_factor_cum_pti=1.0,
            adj_close_pti=120.0,
            asof_timestamp=pd.Timestamp("2021-08-07 16:00"),
        ),
        PITPriceBar(
            trade_date=pd.Timestamp("2021-08-06"),
            ticker="AAPL",
            raw_close=60.0,
            split_factor_cum_pti=2.0,
            dividend_factor_cum_pti=1.0,
            adj_close_pti=120.0,
            asof_timestamp=pd.Timestamp("2021-08-07 16:00"),
        ),
        PITPriceBar(
            trade_date=pd.Timestamp("2021-08-05"),
            ticker="AAPL",
            raw_close=120.0,
            split_factor_cum_pti=1.0,
            dividend_factor_cum_pti=1.0,
            adj_close_pti=120.0,
            asof_timestamp=pd.Timestamp("2021-09-10 16:00"),
        ),
        PITPriceBar(
            trade_date=pd.Timestamp("2021-08-06"),
            ticker="AAPL",
            raw_close=60.0,
            split_factor_cum_pti=2.0,
            dividend_factor_cum_pti=1.0,
            adj_close_pti=120.0,
            asof_timestamp=pd.Timestamp("2021-09-10 16:00"),
        ),
    ]
    engine = InMemoryPITAdjustmentEngine(bars)

    first_split_window = engine.get_adjusted_window(
        "AAPL",
        end_date=pd.Timestamp("2021-08-06"),
        window=2,
        asof=pd.Timestamp("2021-08-07 16:00"),
    )
    chained_window = engine.get_adjusted_window(
        "AAPL",
        end_date=pd.Timestamp("2021-09-10"),
        window=3,
        asof=pd.Timestamp("2021-09-10 16:00"),
    )

    assert not np.isclose(first_split_window.iloc[0], chained_window.iloc[0])
    assert np.isclose(chained_window.iloc[0] / first_split_window.iloc[0], 4.0)
    np.testing.assert_allclose(chained_window.to_numpy(), [960.0, 240.0, 30.0])


def test_pit_no_lookahead_weekly_cutoff() -> None:
    bars = [
        PITPriceBar(
            trade_date=pd.Timestamp("2021-08-06"),
            ticker="AAPL",
            raw_close=120.0,
            split_factor_cum_pti=1.0,
            dividend_factor_cum_pti=1.0,
            adj_close_pti=120.0,
            asof_timestamp=pd.Timestamp("2021-08-06 16:00"),
        ),
        PITPriceBar(
            trade_date=pd.Timestamp("2021-08-06"),
            ticker="AAPL",
            raw_close=120.0,
            split_factor_cum_pti=2.0,
            dividend_factor_cum_pti=1.0,
            adj_close_pti=240.0,
            asof_timestamp=pd.Timestamp("2021-08-09 16:00"),
        ),
    ]
    engine = InMemoryPITAdjustmentEngine(bars)

    prior_friday = engine.get_adj_close(
        "AAPL",
        trade_date=pd.Timestamp("2021-08-06"),
        asof=pd.Timestamp("2021-08-06 16:00"),
    )
    action_visible = engine.get_adj_close(
        "AAPL",
        trade_date=pd.Timestamp("2021-08-06"),
        asof=pd.Timestamp("2021-08-09 16:00"),
    )

    assert prior_friday == 120.0
    assert action_visible == 240.0


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


def test_ledger_pit_chained_split_dividend_compounding() -> None:
    prices = InMemoryRawPriceStore(
        [
            RawPriceObservation(pd.Timestamp("2024-01-02"), "QQQ", 100.0, "official_raw", pd.Timestamp("2024-01-02 16:00")),
            RawPriceObservation(pd.Timestamp("2024-01-03"), "QQQ", 51.0, "official_raw", pd.Timestamp("2024-01-03 16:00")),
            RawPriceObservation(pd.Timestamp("2024-01-04"), "QQQ", 50.0, "official_raw", pd.Timestamp("2024-01-04 16:00")),
        ]
    )
    actions = InMemoryCorporateActionStore(
        [
            CorporateActionEvent("QQQ", "split", pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-02 16:00"), 2.0, "normalized", "official_actions", pd.Timestamp("2024-01-02 16:00")),
            CorporateActionEvent("QQQ", "dividend", pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-03 16:00"), 1.1, "normalized", "official_actions", pd.Timestamp("2024-01-03 16:00")),
        ]
    )
    engine = LedgerPITAdjustmentEngine(raw_price_store=prices, corporate_action_store=actions)

    window = engine.get_adjusted_window("QQQ", pd.Timestamp("2024-01-04"), 3, pd.Timestamp("2024-01-04 16:00"))

    np.testing.assert_allclose(window.to_numpy(), [220.0, 56.1, 50.0])


def test_ledger_pit_weekly_cutoff_no_lookahead() -> None:
    prices = InMemoryRawPriceStore(
        [
            RawPriceObservation(pd.Timestamp("2024-01-05"), "QQQ", 100.0, "official_raw", pd.Timestamp("2024-01-05 16:00")),
        ]
    )
    actions = InMemoryCorporateActionStore(
        [
            CorporateActionEvent("QQQ", "split", pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-08 09:00"), 2.0, "normalized", "official_actions", pd.Timestamp("2024-01-08 09:00")),
        ]
    )
    engine = LedgerPITAdjustmentEngine(raw_price_store=prices, corporate_action_store=actions)

    assert engine.get_adj_close("QQQ", pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-05 16:00")) == 100.0
    assert engine.get_adj_close("QQQ", pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08 16:00")) == 200.0


def test_ledger_pit_rename_crossing_through_identity_resolver() -> None:
    prices = InMemoryRawPriceStore(
        [
            RawPriceObservation(pd.Timestamp("2024-01-30"), "OLD", 100.0, "official_raw", pd.Timestamp("2024-01-30 16:00")),
            RawPriceObservation(pd.Timestamp("2024-01-31"), "OLD", 101.0, "official_raw", pd.Timestamp("2024-01-31 16:00")),
            RawPriceObservation(pd.Timestamp("2024-02-01"), "NEW", 102.0, "official_raw", pd.Timestamp("2024-02-01 16:00")),
        ]
    )
    actions = InMemoryCorporateActionStore([])
    identity = InMemorySymbolIdentityResolver(
        [
            SymbolIdentityRecord("OLD", "NEW", pd.Timestamp("2024-02-01"), "pure_rename", "issuer_actions", pd.Timestamp("2024-01-31 16:00")),
        ]
    )
    engine = LedgerPITAdjustmentEngine(
        raw_price_store=prices,
        corporate_action_store=actions,
        identity_resolver=identity,
    )

    window = engine.get_adjusted_window("NEW", pd.Timestamp("2024-02-01"), 3, pd.Timestamp("2024-02-01 16:00"))

    assert list(window.index) == list(pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01"]))
    np.testing.assert_allclose(window.to_numpy(), [100.0, 101.0, 102.0])


def test_ledger_pit_no_identity_record_fails_closed() -> None:
    prices = InMemoryRawPriceStore(
        [
            RawPriceObservation(pd.Timestamp("2024-01-31"), "OLD", 101.0, "official_raw", pd.Timestamp("2024-01-31 16:00")),
            RawPriceObservation(pd.Timestamp("2024-02-01"), "NEW", 102.0, "official_raw", pd.Timestamp("2024-02-01 16:00")),
        ]
    )
    engine = LedgerPITAdjustmentEngine(
        raw_price_store=prices,
        corporate_action_store=InMemoryCorporateActionStore([]),
    )

    with pytest.raises(DataNotAvailableError):
        engine.get_adjusted_window("NEW", pd.Timestamp("2024-02-01"), 2, pd.Timestamp("2024-02-01 16:00"))


def test_ledger_pit_identity_boundary_no_lookahead() -> None:
    prices = InMemoryRawPriceStore(
        [
            RawPriceObservation(pd.Timestamp("2024-01-31"), "OLD", 101.0, "official_raw", pd.Timestamp("2024-01-31 16:00")),
            RawPriceObservation(pd.Timestamp("2024-02-01"), "NEW", 102.0, "official_raw", pd.Timestamp("2024-02-01 16:00")),
        ]
    )
    identity = InMemorySymbolIdentityResolver(
        [
            SymbolIdentityRecord("OLD", "NEW", pd.Timestamp("2024-02-01"), "pure_rename", "issuer_actions", pd.Timestamp("2024-02-01 09:00")),
        ]
    )
    engine = LedgerPITAdjustmentEngine(
        raw_price_store=prices,
        corporate_action_store=InMemoryCorporateActionStore([]),
        identity_resolver=identity,
    )

    with pytest.raises(DataNotAvailableError):
        engine.get_adjusted_window("NEW", pd.Timestamp("2024-02-01"), 2, pd.Timestamp("2024-01-31 16:00"))


def test_ledger_pit_raw_close_only_reconstruction() -> None:
    prices = InMemoryRawPriceStore(
        [
            RawPriceObservation(pd.Timestamp("2024-01-02"), "QQQ", 10.0, "official_raw", pd.Timestamp("2024-01-02 16:00")),
            RawPriceObservation(pd.Timestamp("2024-01-03"), "QQQ", 11.0, "official_raw", pd.Timestamp("2024-01-03 16:00")),
        ]
    )
    engine = LedgerPITAdjustmentEngine(
        raw_price_store=prices,
        corporate_action_store=InMemoryCorporateActionStore([]),
    )

    window = engine.get_adjusted_window("QQQ", pd.Timestamp("2024-01-03"), 2, pd.Timestamp("2024-01-03 16:00"))

    np.testing.assert_allclose(window.to_numpy(), [10.0, 11.0])
