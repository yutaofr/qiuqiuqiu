from __future__ import annotations

import pandas as pd
import pytest

from qqq_cycle.data_contracts.instrument_namespace import (
    NamespaceNormalizationError,
    normalize_holdings_namespace,
)


def master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "instrument_id": "US0378331005",
                "canonical_symbol": "AAPL",
                "id_type": "ISIN",
                "primary_listing_exchange": "NASDAQ",
                "isin": "US0378331005",
                "cusip": "037833100",
                "sedol": "2046251",
                "share_class": "COMMON",
                "asset_class": "equity",
                "active_from": "1980-12-12",
                "active_to": "",
                "source": "test_master_v1",
            },
            {
                "instrument_id": "US5949181045",
                "canonical_symbol": "MSFT",
                "id_type": "ISIN",
                "primary_listing_exchange": "NASDAQ",
                "isin": "US5949181045",
                "cusip": "594918104",
                "sedol": "2588173",
                "share_class": "COMMON",
                "asset_class": "equity",
                "active_from": "1986-03-13",
                "active_to": "",
                "source": "test_master_v1",
            },
            {
                "instrument_id": "US02079K3059",
                "canonical_symbol": "GOOGL",
                "id_type": "ISIN",
                "primary_listing_exchange": "NASDAQ",
                "isin": "US02079K3059",
                "cusip": "02079K305",
                "sedol": "BYVY8G0",
                "share_class": "A",
                "asset_class": "equity",
                "active_from": "2004-08-19",
                "active_to": "",
                "source": "test_master_v1",
            },
            {
                "instrument_id": "CASH_USD",
                "canonical_symbol": "USD",
                "id_type": "SYNTHETIC",
                "primary_listing_exchange": "",
                "isin": "",
                "cusip": "",
                "sedol": "",
                "share_class": "",
                "asset_class": "cash",
                "active_from": "1900-01-01",
                "active_to": "",
                "source": "test_master_v1",
            },
        ]
    )


def test_exact_isin_cusip_match_resolves() -> None:
    raw = pd.DataFrame([{"raw_symbol": "APPLE", "isin": "US0378331005", "raw_weight": 1.0}])

    normalized, summary = normalize_holdings_namespace(raw, master())

    assert normalized.loc[0, "instrument_id"] == "US0378331005"
    assert normalized.loc[0, "normalization_status"] == "resolved_by_primary_id"
    assert summary.unresolved_weight_sum == 0.0


def test_exact_symbol_exchange_resolves() -> None:
    raw = pd.DataFrame([{"raw_symbol": "MSFT", "exchange": "NASDAQ", "raw_weight": 1.0}])

    normalized, _ = normalize_holdings_namespace(raw, master())

    assert normalized.loc[0, "instrument_id"] == "US5949181045"
    assert normalized.loc[0, "normalization_status"] == "resolved_by_exact_symbol_map"


def test_share_class_map_resolves() -> None:
    raw = pd.DataFrame([{"raw_symbol": "GOOG-A", "exchange": "NASDAQ", "share_class": "A", "raw_weight": 1.0}])
    share_class_map = pd.DataFrame(
        [{"raw_symbol": "GOOG-A", "exchange": "NASDAQ", "share_class": "A", "instrument_id": "US02079K3059"}]
    )

    normalized, _ = normalize_holdings_namespace(raw, master(), share_class_map=share_class_map)

    assert normalized.loc[0, "instrument_id"] == "US02079K3059"
    assert normalized.loc[0, "normalization_status"] == "resolved_by_share_class_map"


def test_override_ledger_resolves_with_override_flag_true() -> None:
    raw = pd.DataFrame([{"raw_symbol": "US DOLLAR", "exchange": "", "share_class": "", "raw_weight": 1.0}])
    override_ledger = pd.DataFrame(
        [{"raw_symbol": "US DOLLAR", "exchange": "", "share_class": "", "instrument_id": "CASH_USD"}]
    )

    normalized, _ = normalize_holdings_namespace(raw, master(), override_ledger=override_ledger)

    assert normalized.loc[0, "instrument_id"] == "CASH_USD"
    assert normalized.loc[0, "normalization_status"] == "resolved_by_override_ledger"
    assert bool(normalized.loc[0, "normalization_override_flag"]) is True


def test_unknown_ticker_becomes_unresolved() -> None:
    raw = pd.DataFrame([{"raw_symbol": "UNKNOWN", "exchange": "NASDAQ", "raw_weight": 1.0}])

    normalized, summary = normalize_holdings_namespace(raw, master())

    assert normalized.loc[0, "normalization_status"] == "unresolved"
    assert summary.unresolved_weight_sum == 1.0


def test_unresolved_weight_sum_uses_weights_not_row_count() -> None:
    raw = pd.DataFrame(
        [
            {"raw_symbol": "AAPL", "exchange": "NASDAQ", "raw_weight": 0.99},
            {"raw_symbol": "UNKNOWN", "exchange": "NASDAQ", "raw_weight": 0.01},
        ]
    )

    _, summary = normalize_holdings_namespace(raw, master())

    assert summary.unresolved_weight_sum == pytest.approx(0.01)
    assert summary.unresolved_weight_blocks is False


def test_unresolved_weight_above_one_percent_blocks_path() -> None:
    raw = pd.DataFrame(
        [
            {"raw_symbol": "AAPL", "exchange": "NASDAQ", "raw_weight": 0.98},
            {"raw_symbol": "UNKNOWN", "exchange": "NASDAQ", "raw_weight": 0.02},
        ]
    )

    _, summary = normalize_holdings_namespace(raw, master())

    assert summary.unresolved_weight_sum == pytest.approx(0.02)
    assert summary.unresolved_weight_blocks is True


def test_fuzzy_name_match_does_not_resolve() -> None:
    raw = pd.DataFrame([{"raw_symbol": "APPLX", "raw_name": "Apple Incorporated", "raw_weight": 1.0}])

    normalized, _ = normalize_holdings_namespace(raw, master())

    assert normalized.loc[0, "normalization_status"] == "unresolved"


def test_raw_ticker_without_canonical_namespace_does_not_resolve() -> None:
    raw = pd.DataFrame([{"raw_symbol": "AAPL", "raw_weight": 1.0}])

    normalized, _ = normalize_holdings_namespace(raw, master())

    assert normalized.loc[0, "normalization_status"] == "unresolved"


def test_weight_unit_unresolved_fails_normalization() -> None:
    raw = pd.DataFrame([{"raw_symbol": "AAPL", "exchange": "NASDAQ", "raw_weight": 42.0}])

    with pytest.raises(NamespaceNormalizationError, match="weight_unit_unresolved"):
        normalize_holdings_namespace(raw, master())
