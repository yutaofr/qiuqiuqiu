"""Fail-closed QQQ holdings weight data contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError


class WeightSumViolationError(Exception):
    """Raised when weight sum is outside tolerance."""


def validate_weight_sum(weights: dict[str, float], tolerance: float = 0.01) -> None:
    """Validate that a retrieved weight vector sums to 1.0 within tolerance.

    Inputs:
        weights: Mapping from ticker to point-in-time disclosed portfolio
            weight.
        tolerance: Maximum absolute distance from 1.0. The default is 0.01.

    Output:
        None when the weight sum is within tolerance.

    Time/as-of semantics:
        This validator does not retrieve data and does not change the as-of
        boundary. It should be called only after `get_weights()` returns a
        timestamp-visible snapshot.

    Failure modes:
        WeightSumViolationError: weight sum is outside `tolerance`.

    Operational note:
        A tolerance greater than 0.01 can hide missing QQQ top weights and
        systematically underestimate c_tau.
    """

    total = sum(float(weight) for weight in weights.values())
    if abs(total - 1.0) > tolerance:
        raise WeightSumViolationError(
            f"weight sum {total:.12g} outside tolerance {tolerance:.12g}"
        )


class WeightStore:
    """Interface for point-in-time holdings weights.

    `get_weights()` is retrieval-only. It must not validate portfolio sum,
    forward-fill, zero-fill, interpolate, or infer missing ticker weights.
    Call `validate_weight_sum()` explicitly when a caller needs a closed
    portfolio-vector check.
    """

    def get_weights(self, trade_date: pd.Timestamp, asof: pd.Timestamp) -> dict[str, float]:
        del trade_date, asof
        raise DataNotAvailableError("weight store is not configured")


class CsvWeightStore(WeightStore):
    """CSV-backed weight store with strict as-of semantics.

    CSV format:
        trade_date,ticker,weight,asof_timestamp
        2021-01-04,AAPL,0.115,2021-01-04T16:00:00
        2021-01-04,MSFT,0.097,2021-01-04T16:00:00

    as-of rule: only rows where asof_timestamp <= asof are visible.

    Retrieval and validation are intentionally decoupled. `get_weights()` only
    returns rows explicitly present for `trade_date` and visible at `asof`; it
    does not perform sum validation, forward-fill missing dates, zero-fill
    absent tickers, or interpolate weights. Use `validate_weight_sum()` for an
    explicit sum check. If tolerance is relaxed above 0.01, missing QQQ top
    weights can systematically bias c_tau downward.
    """

    def __init__(self, path: Path) -> None:
        df = pd.read_csv(path)
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
        df["asof_timestamp"] = pd.to_datetime(df["asof_timestamp"], utc=False)
        df["ticker"] = df["ticker"].str.strip().str.upper()
        df["weight"] = df["weight"].astype(float)
        self._df = df

    def get_weights(self, trade_date: pd.Timestamp, asof: pd.Timestamp) -> dict[str, float]:
        """Return {ticker: weight} visible as of `asof` on `trade_date`.

        Raises DataNotAvailableError if no rows match.
        """
        trade_date = pd.Timestamp(trade_date).normalize()
        asof = pd.Timestamp(asof)
        mask = (self._df["trade_date"] == trade_date) & (self._df["asof_timestamp"] <= asof)
        rows = self._df.loc[mask]
        if rows.empty:
            raise DataNotAvailableError(
                f"no weight data for trade_date={trade_date.date()} asof={asof}"
            )
        return dict(zip(rows["ticker"], rows["weight"]))
