"""Phase 9 symbol identity resolver contract tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError
from qqq_cycle.data_contracts.symbol_identity import CsvSymbolIdentityResolver


def _write_csv(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "identity.csv"
    path.write_text(text.strip() + "\n")
    return path


def test_pure_rename_identity_bridges_old_symbol_when_visible(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        """
old_symbol,new_symbol,effective_date,identity_type,source_label,asof_timestamp
OLD,NEW,2024-02-01,pure_rename,issuer_actions,2024-01-25T16:00:00
""",
    )
    resolver = CsvSymbolIdentityResolver(path)

    assert resolver.resolve_symbol(
        "NEW",
        trade_date=pd.Timestamp("2024-01-31"),
        asof=pd.Timestamp("2024-01-25T16:00:00"),
    ) == "OLD"
    assert resolver.resolve_symbol(
        "NEW",
        trade_date=pd.Timestamp("2024-02-01"),
        asof=pd.Timestamp("2024-02-01T16:00:00"),
    ) == "NEW"


def test_identity_asof_boundary_no_lookahead(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        """
old_symbol,new_symbol,effective_date,identity_type,source_label,asof_timestamp
OLD,NEW,2024-02-01,pure_rename,issuer_actions,2024-01-25T16:00:00
""",
    )
    resolver = CsvSymbolIdentityResolver(path)

    with pytest.raises(DataNotAvailableError):
        resolver.resolve_symbol(
            "NEW",
            trade_date=pd.Timestamp("2024-01-31"),
            asof=pd.Timestamp("2024-01-24T16:00:00"),
        )


def test_non_rename_identity_types_do_not_bridge(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        """
old_symbol,new_symbol,effective_date,identity_type,source_label,asof_timestamp
OLD,NEW,2024-02-01,merger,issuer_actions,2024-01-25T16:00:00
""",
    )
    resolver = CsvSymbolIdentityResolver(path)

    with pytest.raises(DataNotAvailableError):
        resolver.resolve_symbol(
            "NEW",
            trade_date=pd.Timestamp("2024-01-31"),
            asof=pd.Timestamp("2024-01-25T16:00:00"),
        )
