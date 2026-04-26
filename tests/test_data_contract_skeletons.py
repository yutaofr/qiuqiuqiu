import pandas as pd
import pytest

from qqq_cycle.data_contracts.corp_actions import CorporateActionStore
from qqq_cycle.data_contracts.constituents import ConstituentStore
from qqq_cycle.data_contracts.raw_prices import CsvRawPriceStore, RawPriceStore
from qqq_cycle.data_contracts.weights import WeightStore
from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError, HindsightAdjustedDataError


def test_fail_closed_data_contract_stores_raise_clear_errors() -> None:
    asof = pd.Timestamp("2024-01-02 16:00")

    with pytest.raises(DataNotAvailableError):
        RawPriceStore().get_raw_close("QQQ", pd.Timestamp("2024-01-02"), asof)
    with pytest.raises(DataNotAvailableError):
        CorporateActionStore().get_cumulative_factor("QQQ", pd.Timestamp("2024-01-02"), asof)
    with pytest.raises(DataNotAvailableError):
        ConstituentStore().get_snapshot(pd.Timestamp("2024-01-02"), asof)
    with pytest.raises(DataNotAvailableError):
        WeightStore().get_weights(pd.Timestamp("2024-01-02"), asof)


def test_csv_raw_price_store_requires_asof_and_returns_only_visible_raw_close(tmp_path) -> None:
    path = tmp_path / "qqq_raw.csv"
    pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-03"],
            "ticker": ["QQQ", "QQQ"],
            "raw_close": [100.0, 101.0],
            "asof_timestamp": ["2024-01-02 16:05", "2024-01-03 16:05"],
            "source": ["fixture", "fixture"],
        }
    ).to_csv(path, index=False)
    store = CsvRawPriceStore(path)

    assert store.get_raw_close("QQQ", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02 16:05")) == 100.0
    with pytest.raises(DataNotAvailableError):
        store.get_raw_close("QQQ", pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-03 16:04"))


def test_csv_raw_price_store_rejects_hindsight_adjusted_columns(tmp_path) -> None:
    path = tmp_path / "qqq_adjusted.csv"
    pd.DataFrame(
        {
            "trade_date": ["2024-01-02"],
            "ticker": ["QQQ"],
            "adjusted_close": [99.0],
            "asof_timestamp": ["2024-01-02 16:05"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(HindsightAdjustedDataError):
        CsvRawPriceStore(path)
