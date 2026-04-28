"""Fail-closed constituent membership data contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError


@dataclass(frozen=True)
class PITConstituentSnapshot:
    """Point-in-time constituent membership snapshot."""

    trade_date: pd.Timestamp
    members: frozenset[str]
    asof_timestamp: pd.Timestamp


class ConstituentStore:
    """Interface for point-in-time constituent snapshots.

    Implementations retrieve the constituent set that is explicitly recorded
    for `trade_date` and visible as of the caller's decision timestamp. They
    must not carry forward prior membership, silently fill missing dates, or
    substitute related securities.

    Corporate-action semantics:
        Delisting: a delisted ticker is absent from future snapshots unless a
            future row explicitly records it.
        Merger: the disappearing ticker is absent after the merger effective
            date; the surviving/acquiring ticker appears only if its own row is
            present for the requested snapshot.
        Rename: the old symbol terminates and the new symbol is treated as an
            independent member; no automatic bridge is inferred.

    Known limitation:
        Strict no-bridge rename handling can make the micro layer temporarily
        blind to renamed constituents for 20-60 trading days while rolling
        history warms under the new symbol.
    """

    def get_snapshot(
        self, trade_date: pd.Timestamp, asof: pd.Timestamp
    ) -> PITConstituentSnapshot:
        del trade_date, asof
        raise DataNotAvailableError("constituent store is not configured")


class CsvConstituentStore(ConstituentStore):
    """CSV-backed constituent store with strict as-of semantics.

    CSV format (one row per ticker per trade_date):
        trade_date,ticker,asof_timestamp
        2021-01-04,AAPL,2021-01-04T16:00:00
        2021-01-04,MSFT,2021-01-04T16:00:00

    as-of rule: only rows where asof_timestamp <= asof are visible.
    A constituent added Monday (asof=Monday 16:00) is NOT visible to a
    Friday-EOD decision (asof=Friday 16:00) that precedes it.

    The CSV is a snapshot store, not an event resolver. Delists, mergers, and
    renames are represented only by explicit future snapshots:
        - no carry-forward from previous trade dates,
        - no silent fill for missing trade dates,
        - no implicit merger substitution,
        - no automatic old-symbol/new-symbol bridge for renames.

    The rename rule is intentionally conservative but leaves a known open
    production limitation: renamed constituents may be unavailable to micro
    rolling windows for 20-60 trading days after the rename.
    """

    def __init__(self, path: Path) -> None:
        df = pd.read_csv(path)
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
        df["asof_timestamp"] = pd.to_datetime(df["asof_timestamp"], utc=False)
        df["ticker"] = df["ticker"].str.strip().str.upper()
        self._df = df

    def get_snapshot(
        self, trade_date: pd.Timestamp, asof: pd.Timestamp
    ) -> PITConstituentSnapshot:
        """Return the constituent membership visible as of `asof` on `trade_date`.

        Raises DataNotAvailableError if no rows match.
        """
        trade_date = pd.Timestamp(trade_date).normalize()
        asof = pd.Timestamp(asof)
        mask = (self._df["trade_date"] == trade_date) & (self._df["asof_timestamp"] <= asof)
        rows = self._df.loc[mask]
        if rows.empty:
            raise DataNotAvailableError(
                f"no constituent data for trade_date={trade_date.date()} asof={asof}"
            )
        members = frozenset(rows["ticker"].tolist())
        latest_asof = pd.Timestamp(rows["asof_timestamp"].max())
        return PITConstituentSnapshot(
            trade_date=trade_date,
            members=members,
            asof_timestamp=latest_asof,
        )
