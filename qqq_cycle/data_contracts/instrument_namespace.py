"""Canonical instrument namespace normalization for controlled backfills.

Inputs:
    Official holdings rows, a versioned canonical instrument master, optional
    explicit share-class map, and optional explicit override ledger.
Outputs:
    A normalized holdings dataframe with canonical instrument identifiers and
    deterministic normalization statuses.
Time semantics:
    Resolution uses only supplied versioned namespace inputs. It does not use
    price availability, live tickers, or current market state.
As-of semantics:
    Rows are resolved by exact point-in-time identifiers or explicit ledgers;
    fuzzy names and raw ticker joins are intentionally unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd


NORMALIZATION_STATUSES = frozenset(
    {
        "resolved_by_primary_id",
        "resolved_by_exact_symbol_map",
        "resolved_by_share_class_map",
        "resolved_by_override_ledger",
        "unresolved",
        "excluded_non_price_asset",
    }
)

REQUIRED_MASTER_COLUMNS = (
    "instrument_id",
    "canonical_symbol",
    "id_type",
    "primary_listing_exchange",
    "isin",
    "cusip",
    "sedol",
    "share_class",
    "asset_class",
    "active_from",
    "active_to",
    "source",
    "source_version",
)

REQUIRED_NORMALIZED_COLUMNS = (
    "raw_symbol",
    "canonical_symbol",
    "instrument_id",
    "id_type",
    "share_class",
    "raw_weight",
    "normalized_weight",
    "normalization_status",
    "normalization_source",
    "normalization_override_flag",
)


class NamespaceNormalizationError(ValueError):
    pass


@dataclass(frozen=True)
class NormalizationSummary:
    weight_sum: float
    unresolved_weight_sum: float
    unresolved_weight_blocks: bool
    namespace_version_hash: str


def dataframe_sha256(frame: pd.DataFrame) -> str:
    stable = frame.fillna("").astype(str).sort_index(axis=1).to_csv(index=False)
    return sha256(stable.encode("utf-8")).hexdigest()


def canonical_master_hash(master: pd.DataFrame) -> str:
    missing = [col for col in REQUIRED_MASTER_COLUMNS if col not in master.columns]
    if missing:
        raise NamespaceNormalizationError(f"canonical master missing columns: {missing}")
    return dataframe_sha256(master.loc[:, list(REQUIRED_MASTER_COLUMNS)])


def normalize_weight_units(raw_weights: pd.Series) -> pd.Series:
    weights = pd.to_numeric(raw_weights, errors="coerce")
    if weights.isna().any():
        raise NamespaceNormalizationError("weight_unit_unresolved")
    raw_sum = float(weights.sum())
    if 0.99 <= raw_sum <= 1.01:
        return weights.astype(float)
    if 99.0 <= raw_sum <= 101.0:
        return (weights / 100.0).astype(float)
    raise NamespaceNormalizationError("weight_unit_unresolved")


def _clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _clean_upper(value: Any) -> str:
    return _clean(value).upper()


def _first_match(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    return frame.iloc[0]


def _parse_date(value: Any) -> date | None:
    text = _clean(value)
    if not text:
        return None
    return pd.Timestamp(text).date()


def _active_in_asof(row: pd.Series, asof: date | None) -> bool:
    if asof is None:
        return True
    active_from = _parse_date(row.get("active_from"))
    active_to = _parse_date(row.get("active_to"))
    if active_from is not None and asof < active_from:
        return False
    if active_to is not None and asof > active_to:
        return False
    return True


def _filter_active(frame: pd.DataFrame, asof: date | None) -> pd.DataFrame:
    if asof is None or frame.empty:
        return frame
    mask = frame.apply(lambda row: _active_in_asof(row, asof), axis=1)
    return frame.loc[mask]


def _resolve_by_primary_id(row: pd.Series, master: pd.DataFrame, asof: date | None) -> pd.Series | None:
    for raw_col, master_col in (("isin", "isin"), ("cusip", "cusip"), ("sedol", "sedol"), ("figi", "figi")):
        if raw_col not in row.index or master_col not in master.columns:
            continue
        value = _clean_upper(row.get(raw_col))
        if not value:
            continue
        match = _filter_active(master[master[master_col].map(_clean_upper) == value], asof)
        if not match.empty:
            return match.iloc[0]
    return None


def _resolve_by_instrument_id(row: pd.Series, master: pd.DataFrame, asof: date | None) -> pd.Series | None:
    value = _clean(row.get("instrument_id", ""))
    if not value:
        return None
    return _first_match(_filter_active(master[master["instrument_id"].map(_clean) == value], asof))


def _resolve_by_symbol_exchange_share_class(
    row: pd.Series, master: pd.DataFrame, asof: date | None
) -> pd.Series | None:
    symbol = _clean_upper(row.get("raw_symbol", row.get("symbol", "")))
    exchange = _clean_upper(row.get("exchange", ""))
    share_class = _clean_upper(row.get("share_class", ""))
    if not symbol or not exchange:
        return None
    candidate = _filter_active(master[
        (master["canonical_symbol"].map(_clean_upper) == symbol)
        & (master["primary_listing_exchange"].map(_clean_upper) == exchange)
    ], asof)
    if share_class:
        candidate = candidate[candidate["share_class"].map(_clean_upper) == share_class]
    return _first_match(candidate)


def _resolve_from_map(
    row: pd.Series,
    mapping: pd.DataFrame | None,
    *,
    require_active_override: bool = False,
    asof_week_end: str | None = None,
) -> pd.Series | None:
    if mapping is None or mapping.empty:
        return None
    symbol = _clean_upper(row.get("raw_symbol", row.get("symbol", "")))
    exchange = _clean_upper(row.get("exchange", ""))
    share_class = _clean_upper(row.get("share_class", ""))
    raw_symbol_col = "raw_symbol" if "raw_symbol" in mapping.columns else "symbol"
    exchange_col = "raw_exchange" if "raw_exchange" in mapping.columns else (
        "exchange" if "exchange" in mapping.columns else None
    )
    candidate = mapping[mapping[raw_symbol_col].map(_clean_upper) == symbol]
    if exchange_col is not None and exchange:
        candidate = candidate[candidate[exchange_col].map(_clean_upper) == exchange]
    if "share_class" in candidate.columns and share_class:
        candidate = candidate[candidate["share_class"].map(_clean_upper) == share_class]
    if asof_week_end and "effective_from" in candidate.columns:
        asof = pd.Timestamp(asof_week_end).date()
        effective_from = pd.to_datetime(candidate["effective_from"], errors="coerce").dt.date
        effective_to = pd.to_datetime(
            candidate.get("effective_to", pd.Series([""] * len(candidate))), errors="coerce"
        ).dt.date
        candidate = candidate[(effective_from.isna() | (effective_from <= asof))]
        candidate = candidate[(effective_to.isna() | (effective_to >= asof))]
    if require_active_override:
        if "active" in candidate.columns:
            active_mask = candidate["active"].astype(str).str.lower().isin({"1", "true", "yes"})
            candidate = candidate[active_mask]
        if "source_evidence" in candidate.columns:
            candidate = candidate[candidate["source_evidence"].map(_clean).ne("")]
        if asof_week_end and "week_end" in candidate.columns:
            candidate = candidate[candidate["week_end"].map(_clean) == asof_week_end]
    return _first_match(candidate)


def _master_by_instrument_id(master: pd.DataFrame, instrument_id: str, asof: date | None) -> pd.Series | None:
    return _first_match(
        _filter_active(master[master["instrument_id"].map(_clean) == _clean(instrument_id)], asof)
    )


def _resolved_row(
    raw_row: pd.Series,
    master_row: pd.Series,
    normalized_weight: float,
    status: str,
    source: str,
    override_flag: bool,
) -> dict[str, Any]:
    if status not in NORMALIZATION_STATUSES:
        raise NamespaceNormalizationError(f"unknown normalization status: {status}")
    return {
        "raw_symbol": _clean(raw_row.get("raw_symbol", raw_row.get("symbol", ""))),
        "canonical_symbol": _clean(master_row.get("canonical_symbol", "")),
        "instrument_id": _clean(master_row.get("instrument_id", "")),
        "id_type": _clean(master_row.get("id_type", "")),
        "share_class": _clean(master_row.get("share_class", "")),
        "raw_weight": float(raw_row.get("raw_weight", raw_row.get("weight", 0.0))),
        "normalized_weight": float(normalized_weight),
        "normalization_status": status,
        "normalization_source": source,
        "normalization_override_flag": bool(override_flag),
        "raw_name": _clean(raw_row.get("raw_name", raw_row.get("name", ""))),
        "isin": _clean(raw_row.get("isin", "")),
        "cusip": _clean(raw_row.get("cusip", "")),
        "sedol": _clean(raw_row.get("sedol", "")),
        "exchange": _clean(raw_row.get("exchange", "")),
        "asset_class": _clean(master_row.get("asset_class", raw_row.get("asset_class", ""))),
        "price_join_required": bool(master_row.get("price_join_required", True)),
    }


def _unresolved_row(raw_row: pd.Series, normalized_weight: float) -> dict[str, Any]:
    return {
        "raw_symbol": _clean(raw_row.get("raw_symbol", raw_row.get("symbol", ""))),
        "canonical_symbol": "",
        "instrument_id": "",
        "id_type": "",
        "share_class": _clean(raw_row.get("share_class", "")),
        "raw_weight": float(raw_row.get("raw_weight", raw_row.get("weight", 0.0))),
        "normalized_weight": float(normalized_weight),
        "normalization_status": "unresolved",
        "normalization_source": "unresolved",
        "normalization_override_flag": False,
        "raw_name": _clean(raw_row.get("raw_name", raw_row.get("name", ""))),
        "isin": _clean(raw_row.get("isin", "")),
        "cusip": _clean(raw_row.get("cusip", "")),
        "sedol": _clean(raw_row.get("sedol", "")),
        "exchange": _clean(raw_row.get("exchange", "")),
        "asset_class": _clean(raw_row.get("asset_class", "")),
        "price_join_required": True,
    }


def normalize_holdings_namespace(
    raw_holdings: pd.DataFrame,
    canonical_master: pd.DataFrame,
    *,
    share_class_map: pd.DataFrame | None = None,
    override_ledger: pd.DataFrame | None = None,
    unresolved_block_threshold: float = 0.01,
    asof_week_end: str | None = None,
) -> tuple[pd.DataFrame, NormalizationSummary]:
    """Normalize official holdings rows into the canonical instrument namespace."""

    if raw_holdings.empty:
        raise NamespaceNormalizationError("empty_holdings")
    raw_symbol_col = "raw_symbol" if "raw_symbol" in raw_holdings.columns else "symbol"
    weight_col = "raw_weight" if "raw_weight" in raw_holdings.columns else "weight"
    for col in (raw_symbol_col, weight_col):
        if col not in raw_holdings.columns:
            raise NamespaceNormalizationError(f"raw holdings missing column: {col}")
    namespace_hash = canonical_master_hash(canonical_master)

    asof = pd.Timestamp(asof_week_end).date() if asof_week_end else None

    raw = raw_holdings.copy()
    raw["raw_weight"] = pd.to_numeric(raw[weight_col], errors="coerce")
    raw["normalized_weight"] = normalize_weight_units(raw["raw_weight"])

    normalized_rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        normalized_weight = float(row["normalized_weight"])
        master_row = _resolve_by_primary_id(row, canonical_master, asof)
        if master_row is not None:
            normalized_rows.append(
                _resolved_row(
                    row,
                    master_row,
                    normalized_weight,
                    "resolved_by_primary_id",
                    "canonical_master_primary_id",
                    False,
                )
            )
            continue

        master_row = _resolve_by_instrument_id(row, canonical_master, asof)
        if master_row is not None:
            normalized_rows.append(
                _resolved_row(
                    row,
                    master_row,
                    normalized_weight,
                    "resolved_by_primary_id",
                    "canonical_master_instrument_id",
                    False,
                )
            )
            continue

        master_row = _resolve_by_symbol_exchange_share_class(row, canonical_master, asof)
        if master_row is not None:
            normalized_rows.append(
                _resolved_row(
                    row,
                    master_row,
                    normalized_weight,
                    "resolved_by_exact_symbol_map",
                    "canonical_master_symbol_exchange_share_class",
                    False,
                )
            )
            continue

        map_row = _resolve_from_map(row, share_class_map, asof_week_end=asof_week_end)
        if map_row is not None:
            master_row = _master_by_instrument_id(
                canonical_master, str(map_row["instrument_id"]), asof
            )
            if master_row is not None:
                normalized_rows.append(
                    _resolved_row(
                        row,
                        master_row,
                        normalized_weight,
                        "resolved_by_share_class_map",
                        "share_class_map",
                        False,
                    )
                )
                continue

        map_row = _resolve_from_map(
            row,
            override_ledger,
            require_active_override=True,
            asof_week_end=asof_week_end,
        )
        if map_row is not None:
            master_row = _master_by_instrument_id(
                canonical_master, str(map_row["instrument_id"]), asof
            )
            if master_row is not None:
                normalized_rows.append(
                    _resolved_row(
                        row,
                        master_row,
                        normalized_weight,
                        "resolved_by_override_ledger",
                        "instrument_override_ledger",
                        True,
                    )
                )
                continue

        normalized_rows.append(_unresolved_row(row, normalized_weight))

    normalized = pd.DataFrame(normalized_rows)
    normalized = normalized.loc[:, [*REQUIRED_NORMALIZED_COLUMNS, *[c for c in normalized.columns if c not in REQUIRED_NORMALIZED_COLUMNS]]]
    unresolved_weight_sum = float(
        normalized.loc[normalized["normalization_status"] == "unresolved", "normalized_weight"].sum()
    )
    summary = NormalizationSummary(
        weight_sum=float(normalized["normalized_weight"].sum()),
        unresolved_weight_sum=unresolved_weight_sum,
        unresolved_weight_blocks=unresolved_weight_sum > unresolved_block_threshold,
        namespace_version_hash=namespace_hash,
    )
    return normalized, summary


def write_normalized_holdings(path: Path, normalized: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(path, index=False)
