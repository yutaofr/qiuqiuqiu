"""Phase 9 rename identity bridge tests for micro rolling windows."""

from __future__ import annotations

import pandas as pd
import pytest

from qqq_cycle.core.micro_layer import (
    MicroLayerUnavailableError,
    compute_breadth,
    compute_correlation_concentration,
)
from qqq_cycle.data_contracts.corp_actions import InMemoryCorporateActionStore
from qqq_cycle.data_contracts.pit_adjustment import LedgerPITAdjustmentEngine
from qqq_cycle.data_contracts.raw_prices import InMemoryRawPriceStore, RawPriceObservation
from qqq_cycle.data_contracts.symbol_identity import (
    InMemorySymbolIdentityResolver,
    SymbolIdentityRecord,
)


def _rename_engine(*, identity_asof: str | None = "2024-03-25T16:00:00") -> LedgerPITAdjustmentEngine:
    days = pd.bdate_range("2024-01-02", periods=61)
    observations: list[RawPriceObservation] = []
    for i, day in enumerate(days):
        ticker = "OLD" if day < pd.Timestamp("2024-03-26") else "NEW"
        observations.append(
            RawPriceObservation(day, ticker, 100.0 + i, "official_raw", day)
        )
        observations.append(
            RawPriceObservation(day, "B", 200.0 + i * 2.0, "official_raw", day)
        )
    resolver = None
    if identity_asof is not None:
        resolver = InMemorySymbolIdentityResolver(
            [
                SymbolIdentityRecord(
                    "OLD",
                    "NEW",
                    pd.Timestamp("2024-03-26"),
                    "pure_rename",
                    "issuer_actions",
                    pd.Timestamp(identity_asof),
                )
            ]
        )
    return LedgerPITAdjustmentEngine(
        raw_price_store=InMemoryRawPriceStore(observations),
        corporate_action_store=InMemoryCorporateActionStore([]),
        identity_resolver=resolver,
    )


def test_rename_identity_bridge_preserves_breadth_history() -> None:
    engine = _rename_engine()

    breadth = compute_breadth(
        members=frozenset({"NEW"}),
        smoothed_weights={"NEW": 1.0},
        trade_date=pd.Timestamp("2024-03-26"),
        pit_engine=engine,
    )

    assert breadth == pytest.approx(0.0)


def test_rename_identity_bridge_preserves_correlation_history() -> None:
    engine = _rename_engine()
    trade_date = pd.Timestamp("2024-03-26")
    asof = pd.Timestamp("2024-03-26T16:00:00")
    windows = {
        "NEW": engine.get_adjusted_window("NEW", trade_date, 60, asof=asof),
        "B": engine.get_adjusted_window("B", trade_date, 60, asof=asof),
    }

    concentration = compute_correlation_concentration(
        members=frozenset({"NEW", "B"}),
        smoothed_weights={"NEW": 0.5, "B": 0.5},
        price_windows=windows,
    )

    assert concentration == pytest.approx(1.0)


def test_no_identity_record_keeps_fail_closed() -> None:
    engine = _rename_engine(identity_asof=None)

    with pytest.raises(MicroLayerUnavailableError):
        compute_breadth(
            members=frozenset({"NEW"}),
            smoothed_weights={"NEW": 1.0},
            trade_date=pd.Timestamp("2024-03-26"),
            pit_engine=engine,
        )


def test_identity_asof_boundary_no_lookahead() -> None:
    engine = _rename_engine(identity_asof="2024-03-26T09:30:00")

    with pytest.raises(MicroLayerUnavailableError):
        compute_breadth(
            members=frozenset({"NEW"}),
            smoothed_weights={"NEW": 1.0},
            trade_date=pd.Timestamp("2024-03-26"),
            pit_engine=engine,
        )
