import pandas as pd
import pytest

from qqq_cycle.data_contracts.corp_actions import CorporateActionStore
from qqq_cycle.data_contracts.constituents import ConstituentStore
from qqq_cycle.data_contracts.raw_prices import RawPriceStore
from qqq_cycle.data_contracts.weights import WeightStore
from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError


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
