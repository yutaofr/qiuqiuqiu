"""Fail-closed corporate-action data contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError


@dataclass(frozen=True)
class CorporateActionEvent:
    """Normalized point-in-time corporate-action adjustment event.

    All non-standard corporate actions must be normalized upstream into
    equivalent backward adjustment factors before entering the PIT engine. The
    PIT engine only multiplies `equivalent_factor` values; it does not perform
    spin-off, rights issue, odd-lot, or restructuring algebra.
    """

    ticker: str
    event_type: str
    effective_date: pd.Timestamp
    announcement_timestamp: pd.Timestamp
    equivalent_factor: float
    normalization_status: str
    source_label: str
    asof_timestamp: pd.Timestamp


class CorporateActionStore:
    """Interface for PIT normalized corporate-action factors.

    Production inputs are normalized event ledgers. The PIT engine consumes
    only `equivalent_factor` and applies a pure multiplication chain:

        P_adj(tau) = P_raw(tau) * product(f_u for tau < u <= asof_basis)

    Complex corporate-action algebra must happen in ETL before records enter
    this contract.
    """

    def get_cumulative_factor(
        self, ticker: str, trade_date: pd.Timestamp, asof: pd.Timestamp
    ) -> float:
        del ticker, trade_date, asof
        raise DataNotAvailableError("corporate-action store is not configured")

    def get_adjustment_factor(
        self,
        ticker: str,
        start_exclusive: pd.Timestamp,
        end_inclusive: pd.Timestamp,
        asof: pd.Timestamp,
    ) -> float:
        del ticker, start_exclusive, end_inclusive, asof
        raise DataNotAvailableError("corporate-action store is not configured")


class InMemoryCorporateActionStore(CorporateActionStore):
    """Deterministic normalized corporate-action store for contract tests."""

    def __init__(self, events: list[CorporateActionEvent]) -> None:
        self._frame = pd.DataFrame(
            [
                {
                    "ticker": event.ticker.strip().upper(),
                    "event_type": event.event_type,
                    "effective_date": pd.Timestamp(event.effective_date).normalize(),
                    "announcement_timestamp": pd.Timestamp(event.announcement_timestamp),
                    "equivalent_factor": float(event.equivalent_factor),
                    "normalization_status": event.normalization_status,
                    "source_label": event.source_label,
                    "asof_timestamp": pd.Timestamp(event.asof_timestamp),
                }
                for event in events
            ]
        )
        if self._frame.empty:
            self._frame = pd.DataFrame(
                columns=[
                    "ticker",
                    "event_type",
                    "effective_date",
                    "announcement_timestamp",
                    "equivalent_factor",
                    "normalization_status",
                    "source_label",
                    "asof_timestamp",
                ]
            )
        self._frame = self._frame.sort_values(["ticker", "effective_date", "asof_timestamp"])
        self._empty = self._frame.empty

    def get_cumulative_factor(
        self, ticker: str, trade_date: pd.Timestamp, asof: pd.Timestamp
    ) -> float:
        return self.get_adjustment_factor(
            ticker,
            start_exclusive=pd.Timestamp.min,
            end_inclusive=trade_date,
            asof=asof,
        )

    def get_adjustment_factor(
        self,
        ticker: str,
        start_exclusive: pd.Timestamp,
        end_inclusive: pd.Timestamp,
        asof: pd.Timestamp,
    ) -> float:
        start_ts = pd.Timestamp(start_exclusive).normalize()
        end_ts = pd.Timestamp(end_inclusive).normalize()
        asof_ts = pd.Timestamp(asof)
        if self._empty:
            return 1.0
        rows = self._frame[
            (self._frame["ticker"] == ticker.strip().upper())
            & (self._frame["effective_date"] > start_ts)
            & (self._frame["effective_date"] <= end_ts)
            & (self._frame["asof_timestamp"] <= asof_ts)
            & (self._frame["normalization_status"] == "normalized")
        ]
        if rows.empty:
            return 1.0
        return float(rows["equivalent_factor"].prod())


class CsvCorporateActionStore(InMemoryCorporateActionStore):
    """CSV-backed normalized corporate-action ledger.

    Required columns:
        ticker,event_type,effective_date,announcement_timestamp,
        equivalent_factor,normalization_status,source_label,asof_timestamp
    """

    REQUIRED = {
        "ticker",
        "event_type",
        "effective_date",
        "announcement_timestamp",
        "equivalent_factor",
        "normalization_status",
        "source_label",
        "asof_timestamp",
    }

    def __init__(self, path: str | Path) -> None:
        raw = pd.read_csv(Path(path))
        normalized = {str(col).strip().lower(): col for col in raw.columns}
        missing = self.REQUIRED.difference(normalized)
        if missing:
            raise DataNotAvailableError(
                f"corporate-action CSV missing required columns: {sorted(missing)}"
            )
        frame = raw.rename(columns={original: key for key, original in normalized.items()})
        super().__init__(
            [
                CorporateActionEvent(
                    ticker=str(row["ticker"]),
                    event_type=str(row["event_type"]),
                    effective_date=pd.Timestamp(row["effective_date"]),
                    announcement_timestamp=pd.Timestamp(row["announcement_timestamp"]),
                    equivalent_factor=float(row["equivalent_factor"]),
                    normalization_status=str(row["normalization_status"]),
                    source_label=str(row["source_label"]),
                    asof_timestamp=pd.Timestamp(row["asof_timestamp"]),
                )
                for _, row in frame.iterrows()
            ]
        )
