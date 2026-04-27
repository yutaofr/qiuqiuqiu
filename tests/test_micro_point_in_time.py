import pandas as pd

from qqq_cycle.core.micro_layer import compute_breadth


class _AsOfRecordingPITEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, pd.Timestamp, int, pd.Timestamp]] = []

    def get_adjusted_window(
        self, ticker: str, end_date: pd.Timestamp, window: int, asof: pd.Timestamp
    ) -> pd.Series:
        self.calls.append((ticker, pd.Timestamp(end_date), window, pd.Timestamp(asof)))
        idx = pd.date_range(end=pd.Timestamp(end_date), periods=window, freq="B")
        return pd.Series([100.0] * (window - 1) + [101.0], index=idx)


def test_micro_breadth_uses_trade_date_as_pit_asof() -> None:
    trade_date = pd.Timestamp("2024-03-29")
    engine = _AsOfRecordingPITEngine()

    compute_breadth(
        members=frozenset({"A", "B"}),
        smoothed_weights={"A": 0.5, "B": 0.5},
        trade_date=trade_date,
        pit_engine=engine,
    )

    assert engine.calls == [
        ("A", trade_date, 20, trade_date),
        ("B", trade_date, 20, trade_date),
    ]
