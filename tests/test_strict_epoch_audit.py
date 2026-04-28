"""Phase 9 strict production epoch audit tests."""

from __future__ import annotations

import pandas as pd
import pytest

from qqq_cycle.backtest.strict_epoch_audit import derive_production_strict_epoch
from qqq_cycle.data_contracts.constituents import CsvConstituentStore
from qqq_cycle.data_contracts.corp_actions import InMemoryCorporateActionStore
from qqq_cycle.data_contracts.pit_adjustment import LedgerPITAdjustmentEngine
from qqq_cycle.data_contracts.raw_prices import InMemoryRawPriceStore, RawPriceObservation
from qqq_cycle.data_contracts.weights import CsvWeightStore


def _store_from_csv(store_cls, text: str):
    import io

    path_like = io.StringIO(text.strip())
    df = pd.read_csv(path_like)
    store = store_cls.__new__(store_cls)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    df["asof_timestamp"] = pd.to_datetime(df["asof_timestamp"], utc=False)
    df["ticker"] = df["ticker"].str.strip().str.upper()
    if "weight" in df:
        df["weight"] = df["weight"].astype(float)
    store._df = df
    return store


def _raw_prices(days: pd.DatetimeIndex, tickers: tuple[str, ...]) -> InMemoryRawPriceStore:
    return InMemoryRawPriceStore(
        [
            RawPriceObservation(day, ticker, 100.0 + i, "official_raw", day + pd.Timedelta(hours=16))
            for i, day in enumerate(days)
            for ticker in tickers
        ]
    )


def test_epoch_start_derivation_uses_first_fully_covered_date() -> None:
    days = pd.bdate_range("2024-01-01", periods=5)
    constituents = _store_from_csv(
        CsvConstituentStore,
        """
trade_date,ticker,asof_timestamp
2024-01-01,A,2024-01-01T16:00:00
2024-01-01,B,2024-01-01T16:00:00
2024-01-02,A,2024-01-02T16:00:00
2024-01-02,B,2024-01-02T16:00:00
2024-01-03,A,2024-01-03T16:00:00
2024-01-03,B,2024-01-03T16:00:00
2024-01-04,A,2024-01-04T16:00:00
2024-01-04,B,2024-01-04T16:00:00
2024-01-05,A,2024-01-05T16:00:00
2024-01-05,B,2024-01-05T16:00:00
""",
    )
    weights = _store_from_csv(
        CsvWeightStore,
        """
trade_date,ticker,weight,asof_timestamp
2024-01-01,A,0.50,2024-01-01T16:00:00
2024-01-01,B,0.50,2024-01-01T16:00:00
2024-01-02,A,0.50,2024-01-02T16:00:00
2024-01-02,B,0.50,2024-01-02T16:00:00
2024-01-03,A,0.50,2024-01-03T16:00:00
2024-01-03,B,0.50,2024-01-03T16:00:00
2024-01-04,A,0.50,2024-01-04T16:00:00
2024-01-04,B,0.50,2024-01-04T16:00:00
2024-01-05,A,0.50,2024-01-05T16:00:00
2024-01-05,B,0.50,2024-01-05T16:00:00
""",
    )
    pit = LedgerPITAdjustmentEngine(
        raw_price_store=_raw_prices(days, ("A", "B")),
        corporate_action_store=InMemoryCorporateActionStore([]),
    )

    manifest = derive_production_strict_epoch(
        days,
        pit_engine=pit,
        constituent_store=constituents,
        weight_store=weights,
        pit_window=3,
    )

    assert manifest.production_strict_epoch_start == "2024-01-03"
    assert manifest.constituent_coverage_ok is True
    assert manifest.weight_coverage_ok is True
    assert manifest.pit_coverage_ok is True
    assert manifest.open_blockers == []


def test_no_forward_fill_across_epoch_boundary() -> None:
    days = pd.bdate_range("2024-01-01", periods=4)
    constituents = _store_from_csv(
        CsvConstituentStore,
        """
trade_date,ticker,asof_timestamp
2024-01-01,A,2024-01-01T16:00:00
2024-01-02,A,2024-01-02T16:00:00
2024-01-04,A,2024-01-04T16:00:00
""",
    )
    weights = _store_from_csv(
        CsvWeightStore,
        """
trade_date,ticker,weight,asof_timestamp
2024-01-01,A,1.0,2024-01-01T16:00:00
2024-01-02,A,1.0,2024-01-02T16:00:00
2024-01-03,A,1.0,2024-01-03T16:00:00
2024-01-04,A,1.0,2024-01-04T16:00:00
""",
    )
    pit = LedgerPITAdjustmentEngine(
        raw_price_store=_raw_prices(days, ("A",)),
        corporate_action_store=InMemoryCorporateActionStore([]),
    )

    manifest = derive_production_strict_epoch(
        days,
        pit_engine=pit,
        constituent_store=constituents,
        weight_store=weights,
        pit_window=2,
    )

    assert manifest.production_strict_epoch_start == "2024-01-04"
    assert manifest.row_modes["2024-01-03"] == "degraded_by_design"
    assert manifest.row_modes["2024-01-04"] == "strict_eligible"
    assert manifest.open_blockers == []


def test_pre_epoch_rows_degrade_and_post_epoch_rows_are_eligible() -> None:
    days = pd.bdate_range("2024-01-01", periods=4)
    constituents = _store_from_csv(
        CsvConstituentStore,
        "\n".join(
            ["trade_date,ticker,asof_timestamp"]
            + [f"{day.date()},A,{day.date()}T16:00:00" for day in days]
        ),
    )
    weights = _store_from_csv(
        CsvWeightStore,
        "\n".join(
            ["trade_date,ticker,weight,asof_timestamp"]
            + [f"{day.date()},A,1.0,{day.date()}T16:00:00" for day in days]
        ),
    )
    pit = LedgerPITAdjustmentEngine(
        raw_price_store=_raw_prices(days, ("A",)),
        corporate_action_store=InMemoryCorporateActionStore([]),
    )

    manifest = derive_production_strict_epoch(
        days,
        pit_engine=pit,
        constituent_store=constituents,
        weight_store=weights,
        pit_window=2,
    )

    assert manifest.row_modes["2024-01-01"] == "degraded_by_design"
    assert manifest.row_modes["2024-01-02"] == "strict_eligible"
    assert manifest.row_modes["2024-01-04"] == "strict_eligible"
