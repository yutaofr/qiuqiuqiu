"""Point-in-time symbol identity resolver for pure rename continuity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError


@dataclass(frozen=True)
class SymbolIdentityRecord:
    """Explicit symbol identity mapping.

    Only `identity_type == "pure_rename"` may bridge price history. Mergers,
    spin-offs, and other restructurings are not automatically treated as the
    same economic entity.
    """

    old_symbol: str
    new_symbol: str
    effective_date: pd.Timestamp
    identity_type: str
    source_label: str
    asof_timestamp: pd.Timestamp


class SymbolIdentityResolver:
    """Resolve a requested symbol to its valid historical symbol as of time."""

    def resolve_symbol(
        self, ticker: str, trade_date: pd.Timestamp, asof: pd.Timestamp
    ) -> str:
        del ticker, trade_date, asof
        raise DataNotAvailableError("symbol identity resolver is not configured")


class InMemorySymbolIdentityResolver(SymbolIdentityResolver):
    """In-memory resolver with strict as-of and pure-rename-only semantics."""

    def __init__(self, records: list[SymbolIdentityRecord]) -> None:
        self._frame = pd.DataFrame(
            [
                {
                    "old_symbol": record.old_symbol.strip().upper(),
                    "new_symbol": record.new_symbol.strip().upper(),
                    "effective_date": pd.Timestamp(record.effective_date).normalize(),
                    "identity_type": record.identity_type,
                    "source_label": record.source_label,
                    "asof_timestamp": pd.Timestamp(record.asof_timestamp),
                }
                for record in records
            ]
        )
        if self._frame.empty:
            self._frame = pd.DataFrame(
                columns=[
                    "old_symbol",
                    "new_symbol",
                    "effective_date",
                    "identity_type",
                    "source_label",
                    "asof_timestamp",
                ]
            )
        self._frame = self._frame.sort_values(["new_symbol", "effective_date", "asof_timestamp"])

    def resolve_symbol(
        self, ticker: str, trade_date: pd.Timestamp, asof: pd.Timestamp
    ) -> str:
        requested = ticker.strip().upper()
        trade_ts = pd.Timestamp(trade_date).normalize()
        asof_ts = pd.Timestamp(asof)
        if self._frame.empty:
            return requested
        candidates = self._frame[
            (self._frame["new_symbol"] == requested)
            & (self._frame["effective_date"] > trade_ts)
            & (self._frame["asof_timestamp"] <= asof_ts)
        ]
        if candidates.empty:
            future = self._frame[
                (self._frame["new_symbol"] == requested)
                & (self._frame["effective_date"] > trade_ts)
            ]
            if future.empty:
                return requested
            raise DataNotAvailableError(
                f"identity record for {requested} before {future.iloc[0]['effective_date']} "
                f"not visible as of {asof_ts}"
            )
        rec = candidates.iloc[-1]
        if rec["identity_type"] != "pure_rename":
            raise DataNotAvailableError(
                f"identity type {rec['identity_type']} is not bridgeable for {requested}"
            )
        return str(rec["old_symbol"])


class CsvSymbolIdentityResolver(InMemorySymbolIdentityResolver):
    """CSV-backed symbol identity resolver.

    Required columns:
        old_symbol,new_symbol,effective_date,identity_type,source_label,asof_timestamp
    """

    REQUIRED = {
        "old_symbol",
        "new_symbol",
        "effective_date",
        "identity_type",
        "source_label",
        "asof_timestamp",
    }

    def __init__(self, path: str | Path) -> None:
        raw = pd.read_csv(Path(path))
        normalized = {str(col).strip().lower(): col for col in raw.columns}
        missing = self.REQUIRED.difference(normalized)
        if missing:
            raise DataNotAvailableError(
                f"symbol identity CSV missing required columns: {sorted(missing)}"
            )
        frame = raw.rename(columns={original: key for key, original in normalized.items()})
        super().__init__(
            [
                SymbolIdentityRecord(
                    old_symbol=str(row["old_symbol"]),
                    new_symbol=str(row["new_symbol"]),
                    effective_date=pd.Timestamp(row["effective_date"]),
                    identity_type=str(row["identity_type"]),
                    source_label=str(row["source_label"]),
                    asof_timestamp=pd.Timestamp(row["asof_timestamp"]),
                )
                for _, row in frame.iterrows()
            ]
        )
