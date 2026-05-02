"""Live contract resolution for controlled backfill weeks.

Inputs:
    Controlled backfill result artifacts, strict/backfill store roots, and the
    legacy cache/micro directory for non-controlled weeks.
Outputs:
    PipelineContracts plus auditable routing metadata.
Time semantics:
    Controlled backfill result is the highest-priority control input for its
    week. Legacy cache/micro is never consulted for controlled block or
    degraded_backfill weeks.
As-of semantics:
    strict_recovery must pass the strict store gate before any strict contract
    is returned. The resolver never treats file existence alone as strict
    legality.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from qqq_cycle.data_contracts.constituents import CsvConstituentStore
from qqq_cycle.data_contracts.corp_actions import InMemoryCorporateActionStore
from qqq_cycle.data_contracts.pit_adjustment import LedgerPITAdjustmentEngine
from qqq_cycle.data_contracts.raw_prices import CsvRawPriceStore
from qqq_cycle.data_contracts.symbol_identity import InMemorySymbolIdentityResolver
from qqq_cycle.data_contracts.weights import CsvWeightStore
from qqq_cycle.ops.backfill_ingest import evaluate_strict_store_gate
from qqq_cycle.pipeline import PipelineContracts


ContractSource = Literal[
    "stores_strict",
    "stores_backfill",
    "controlled_block",
    "legacy_cache_micro",
    "none",
]


@dataclass(frozen=True)
class LiveContractResolution:
    contracts: PipelineContracts | None
    backfill_mode: str | None
    contract_source: ContractSource
    strict_gate_passed: bool
    reason: str


def resolve_live_contracts_for_week(
    *,
    week_end: str,
    asset: str,
    controlled_backfill_result: dict | None,
    store_root: str | Path,
    legacy_cache_micro_dir: str | Path,
    production_ledger_dir: str | Path,
) -> LiveContractResolution:
    """Resolve live contracts for one week with controlled result priority."""

    if controlled_backfill_result is None:
        legacy = _build_legacy_cache_micro_contracts(
            legacy_cache_micro_dir=Path(legacy_cache_micro_dir),
            production_ledger_dir=Path(production_ledger_dir),
        )
        if legacy is None:
            return LiveContractResolution(None, None, "none", False, "legacy_cache_micro_missing")
        return LiveContractResolution(
            legacy,
            None,
            "legacy_cache_micro",
            False,
            "legacy_cache_micro_used_for_non_controlled_week",
        )

    mode = str(controlled_backfill_result.get("backfill_mode", "block"))
    if mode == "strict_recovery":
        gate = evaluate_strict_store_gate(week_end=week_end, asset=asset, store_root=store_root)
        if not gate.passed:
            return LiveContractResolution(
                None,
                "block",
                "controlled_block",
                False,
                f"strict_store_gate_failed:{gate.reason}",
            )
        contracts = _build_store_strict_contracts(
            week_end=week_end,
            asset=asset,
            store_root=Path(store_root),
        )
        if contracts is None:
            return LiveContractResolution(
                None,
                "strict_recovery",
                "stores_strict",
                True,
                "strict_store_gate_passed_but_no_pit_contract",
            )
        return LiveContractResolution(
            contracts,
            "strict_recovery",
            "stores_strict",
            True,
            "strict_store_gate_passed",
        )

    if mode == "degraded_backfill":
        return LiveContractResolution(
            None,
            "degraded_backfill",
            "stores_backfill",
            False,
            "controlled_degraded_backfill_freezes_micro",
        )

    return LiveContractResolution(
        None,
        "block",
        "controlled_block",
        False,
        "controlled_block_prevents_strict_contracts",
    )


def _build_store_strict_contracts(
    *,
    week_end: str,
    asset: str,
    store_root: Path,
) -> PipelineContracts | None:
    """Build strict contracts only from stores/strict artifacts."""

    asset_lc = asset.lower()
    weekly_h_t_path = store_root / "strict" / "weekly_h_t" / f"{asset_lc}_weekly_h_t_{week_end}.csv"
    if weekly_h_t_path.exists():
        frame = pd.read_csv(weekly_h_t_path)
        if not {"week_end", "h_t"}.issubset(frame.columns):
            return None
        series = pd.Series(
            pd.to_numeric(frame["h_t"], errors="coerce").to_numpy(dtype=float),
            index=pd.to_datetime(frame["week_end"]),
            name="h_t",
        )
        return PipelineContracts(
            weekly_h_t=series,
            pit_engine_available=True,
            constituents_available=True,
            weights_available=True,
        )

    constituents_dir = store_root / "strict" / "constituents"
    weights_dir = store_root / "strict" / "weights"
    prices_dir = store_root / "strict" / "prices"
    if not constituents_dir.exists() or not weights_dir.exists() or not prices_dir.exists():
        return None

    price_rows: list[pd.DataFrame] = []
    for path in sorted(prices_dir.glob("*.csv")):
        frame = pd.read_csv(path, usecols=["trade_date", "raw_close", "asof_timestamp"])
        frame["ticker"] = path.stem.upper()
        frame["source_label"] = "strict_store_raw_close"
        price_rows.append(frame[["trade_date", "ticker", "raw_close", "source_label", "asof_timestamp"]])
    if not price_rows:
        return None

    ledger_dir = store_root / "strict" / "production_ledgers"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_dir / f"{asset_lc}_raw_prices_{week_end}.csv"
    pd.concat(price_rows, ignore_index=True).sort_values(["ticker", "trade_date"]).to_csv(
        ledger_path, index=False
    )
    pit_engine = LedgerPITAdjustmentEngine(
        raw_price_store=CsvRawPriceStore(ledger_path),
        corporate_action_store=InMemoryCorporateActionStore([]),
        identity_resolver=InMemorySymbolIdentityResolver([]),
    )
    # Aggregate all historical constituents and weights from their respective directories
    const_rows: list[pd.DataFrame] = []
    for path in sorted(constituents_dir.glob(f"{asset_lc}_constituents_*.csv")):
        const_rows.append(pd.read_csv(path))
    if not const_rows:
        return None
    
    weight_rows: list[pd.DataFrame] = []
    for path in sorted(weights_dir.glob(f"{asset_lc}_weights_*.csv")):
        weight_rows.append(pd.read_csv(path))
    if not weight_rows:
        return None

    # Write unified historical files for the session
    tmp_const_path = store_root / "strict" / f"tmp_all_constituents_{week_end}.csv"
    tmp_weight_path = store_root / "strict" / f"tmp_all_weights_{week_end}.csv"
    pd.concat(const_rows, ignore_index=True).to_csv(tmp_const_path, index=False)
    pd.concat(weight_rows, ignore_index=True).to_csv(tmp_weight_path, index=False)

    return PipelineContracts(
        pit_engine=pit_engine,
        constituent_store=CsvConstituentStore(tmp_const_path),
        weight_store=CsvWeightStore(tmp_weight_path),
    )


def _build_legacy_cache_micro_contracts(
    *,
    legacy_cache_micro_dir: Path,
    production_ledger_dir: Path,
) -> PipelineContracts | None:
    """Build legacy cache/micro contracts for non-controlled weeks only."""

    const_csv = legacy_cache_micro_dir / "constituents.csv"
    weights_csv = legacy_cache_micro_dir / "weights.csv"
    prices_dir = legacy_cache_micro_dir / "prices"
    if not const_csv.exists() or not weights_csv.exists():
        return None

    rows: list[pd.DataFrame] = []
    for path in sorted(prices_dir.glob("*.csv")):
        frame = pd.read_csv(path, usecols=["trade_date", "raw_close", "asof_timestamp"])
        frame["ticker"] = path.stem.upper()
        frame["source_label"] = "local_seed_raw_close"
        rows.append(frame[["trade_date", "ticker", "raw_close", "source_label", "asof_timestamp"]])
    if not rows:
        return None

    production_ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = production_ledger_dir / "raw_prices.csv"
    pd.concat(rows, ignore_index=True).sort_values(["ticker", "trade_date"]).to_csv(
        ledger_path, index=False
    )
    pit_engine = LedgerPITAdjustmentEngine(
        raw_price_store=CsvRawPriceStore(ledger_path),
        corporate_action_store=InMemoryCorporateActionStore([]),
        identity_resolver=InMemorySymbolIdentityResolver([]),
    )
    return PipelineContracts(
        pit_engine=pit_engine,
        constituent_store=CsvConstituentStore(const_csv),
        weight_store=CsvWeightStore(weights_csv),
    )
