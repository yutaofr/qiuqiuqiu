import json
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import pytest

from qqq_cycle.backtest.real_replay import RealReplayConfig, run_real_replay
from qqq_cycle.data_contracts.macro_prices import (
    CsvMacroMarketPriceStore,
    MacroMarketPriceContract,
    PriceBasis,
)
from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError, degrade_micro_mode


def test_real_replay_degrades_when_required_official_sources_missing(tmp_path: Path) -> None:
    config = RealReplayConfig(
        cache_root=tmp_path / "cache",
        output_dir=tmp_path / "outputs",
        fred_series=("DFII10",),
        fetch_fred=False,
        fetch_ai_gpr=False,
        qqq_price_csv=None,
    )

    result = run_real_replay(config)

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["mode"] == "degraded"
    assert "QQQ" in manifest["missing_sources"]
    assert "AI_GPR" in manifest["missing_sources"]
    assert (tmp_path / "cache" / "raw").exists()
    assert (tmp_path / "cache" / "staging").exists()
    assert (tmp_path / "cache" / "manifests").exists()
    assert result.weekly_replay_path is None
    assert (result.output_dir / "missing_sources.json").exists()


def test_real_replay_degrades_when_qqq_csv_is_not_macro_price_data(tmp_path: Path) -> None:
    qqq_csv = tmp_path / "qqq_bad.csv"
    qqq_csv.write_text("date,adjusted_close\n2024-01-02,99.0\n", encoding="utf-8")
    config = RealReplayConfig(
        cache_root=tmp_path / "cache",
        output_dir=tmp_path / "outputs",
        fred_series=(),
        fetch_fred=False,
        fetch_ai_gpr=False,
        qqq_price_csv=qqq_csv,
    )

    result = run_real_replay(config)

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["mode"] == "degraded"
    assert "QQQ" in manifest["missing_sources"]
    assert "missing required columns" in manifest["source_errors"]["QQQ"].lower()


def test_macro_market_price_contract_loads_vendor_adjusted_close_without_pit_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qqq_macro.csv"
    pd.DataFrame(
        {
            "trade_date": ["2024-01-05", "2024-01-12"],
            "ticker": ["QQQ", "QQQ"],
            "close": [400.0, 405.0],
            "source_name": ["diagnostic_vendor", "diagnostic_vendor"],
            "fetch_timestamp": ["2024-01-12 17:00", "2024-01-12 17:00"],
            "price_basis": [
                "vendor_backward_adjusted",
                "vendor_backward_adjusted",
            ],
        }
    ).to_csv(path, index=False)

    store = CsvMacroMarketPriceStore(path)
    series = store.to_series("QQQ", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31"))

    assert list(series.to_numpy()) == [400.0, 405.0]
    assert store.price_basis == PriceBasis.VENDOR_BACKWARD_ADJUSTED


def test_macro_market_price_contract_is_forbidden_for_micro_paths(tmp_path: Path) -> None:
    contract = MacroMarketPriceContract(
        trade_date=pd.Timestamp("2024-01-05"),
        ticker="QQQ",
        close=400.0,
        source_name="diagnostic_vendor",
        fetch_timestamp=pd.Timestamp("2024-01-05 17:00"),
        price_basis=PriceBasis.OFFICIAL_MARKET_CLOSE,
    )

    with pytest.raises(DataNotAvailableError, match="MacroMarketPriceContract"):
        degrade_micro_mode(contract)  # type: ignore[arg-type]


def test_real_replay_uses_compliant_macro_qqq_for_state_stress_only_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = pd.date_range("2000-01-07", "2025-12-26", freq="W-FRI")
    trend = np.linspace(0.0, 1.0, len(idx))
    series_map = {
        "DFII10": pd.Series(0.4 + 0.2 * np.sin(trend * 20), index=idx),
        "DGS2": pd.Series(2.0 + 0.5 * np.sin(trend * 12), index=idx),
        "BAMLH0A0HYM2": pd.Series(4.0 + 0.4 * np.cos(trend * 14), index=idx),
        "NFCI": pd.Series(-0.25 + 0.2 * np.sin(trend * 17), index=idx),
        "VIXCLS": pd.Series(18.0 + 4.0 * np.sin(trend * 19), index=idx),
        "USEPUINDXD": pd.Series(90.0 + 20.0 * np.cos(trend * 15), index=idx),
    }
    ai_gpr = pd.Series(50.0 + 15.0 * np.sin(trend * 18), index=idx)
    qqq = 100.0 * np.exp(np.cumsum(0.002 + 0.01 * np.sin(trend * 16)))
    qqq_csv = tmp_path / "qqq_macro.csv"
    pd.DataFrame(
        {
            "trade_date": idx,
            "ticker": "QQQ",
            "close": qqq,
            "source_name": "diagnostic_vendor",
            "fetch_timestamp": "2026-04-27 00:00",
            "price_basis": "vendor_backward_adjusted",
        }
    ).to_csv(qqq_csv, index=False)

    def fake_fetch(series_id: str, api_key: str, start: str, end: str) -> pd.Series:
        del api_key, start, end
        return series_map[series_id]

    monkeypatch.setenv("FRED_API_KEY", "test-key")
    monkeypatch.setattr("qqq_cycle.backtest.real_replay.fetch_fred_series", fake_fetch)
    monkeypatch.setattr("qqq_cycle.backtest.real_replay._load_ai_gpr", lambda *_: ai_gpr)

    result = run_real_replay(
        RealReplayConfig(
            cache_root=tmp_path / "cache",
            output_dir=tmp_path / "outputs",
            qqq_price_csv=qqq_csv,
        )
    )

    manifest = json.loads(result.manifest_path.read_text())
    metadata = json.loads((result.output_dir / "metadata.json").read_text())
    weekly = pd.read_csv(result.weekly_replay_path)

    assert result.mode == "state_stress_only"
    assert result.weekly_replay_path is not None
    assert manifest["replay_scope"] == "state_stress_only"
    assert manifest["micro_scope"] == "disabled_no_pit_micro_contract"
    assert manifest["risk_scope"] == "disabled_no_production_rho_t"
    assert manifest["price_basis"] == "vendor_backward_adjusted"
    assert manifest["missing_sources"] == []
    assert not (result.output_dir / "missing_sources.json").exists()
    assert set(metadata["active_sources"]) == set(
        ["DFII10", "DGS2", "BAMLH0A0HYM2", "NFCI", "VIXCLS", "USEPUINDXD", "AI_GPR", "QQQ"]
    )
    assert {
        "week_end",
        "L_t",
        "T_t",
        "P_t",
        "E_t",
        "H_t",
        "I_t",
        "state_probs_json",
        "state_label",
        "d_t",
        "a_t",
        "s_t",
        "drift_probe_raw",
        "drift_flag",
    }.issubset(weekly.columns)
    for name in ("2008_09_to_2009_06", "2020_02_to_2020_06", "2021_10_to_2022_03"):
        assert (result.output_dir / f"event_{name}.csv").exists()


def test_real_replay_degrades_gracefully_when_replay_warmup_is_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = pd.date_range("2024-01-05", periods=40, freq="W-FRI")
    series = pd.Series(np.linspace(1.0, 2.0, len(idx)), index=idx)
    qqq_csv = tmp_path / "qqq_macro.csv"
    pd.DataFrame(
        {
            "trade_date": idx,
            "ticker": "QQQ",
            "close": np.linspace(400.0, 420.0, len(idx)),
            "source_name": "diagnostic_vendor",
            "fetch_timestamp": "2026-04-27 00:00",
            "price_basis": "official_market_close",
        }
    ).to_csv(qqq_csv, index=False)

    monkeypatch.setenv("FRED_API_KEY", "test-key")
    monkeypatch.setattr(
        "qqq_cycle.backtest.real_replay.fetch_fred_series",
        lambda *_: series,
    )
    monkeypatch.setattr("qqq_cycle.backtest.real_replay._load_ai_gpr", lambda *_: series)

    result = run_real_replay(
        RealReplayConfig(
            cache_root=tmp_path / "cache",
            output_dir=tmp_path / "outputs",
            qqq_price_csv=qqq_csv,
        )
    )

    manifest = json.loads(result.manifest_path.read_text())
    assert result.mode == "degraded"
    assert result.weekly_replay_path is None
    assert "REPLAY_WARMUP" in manifest["missing_sources"]
    assert "need at least 265 finite theta rows" in manifest["source_errors"]["REPLAY_WARMUP"]


def test_real_replay_accepts_series_csv_override_for_historical_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = pd.date_range("2000-01-07", "2025-12-26", freq="W-FRI")
    trend = np.linspace(0.0, 1.0, len(idx))
    live_map = {
        "DFII10": pd.Series(0.4 + 0.2 * np.sin(trend * 20), index=idx),
        "DGS2": pd.Series(2.0 + 0.5 * np.sin(trend * 12), index=idx),
        "NFCI": pd.Series(-0.25 + 0.2 * np.sin(trend * 17), index=idx),
        "VIXCLS": pd.Series(18.0 + 4.0 * np.sin(trend * 19), index=idx),
        "USEPUINDXD": pd.Series(90.0 + 20.0 * np.cos(trend * 15), index=idx),
    }
    archive_csv = tmp_path / "BAMLH0A0HYM2.csv"
    pd.DataFrame(
        {
            "DATE": idx[:900],
            "BAMLH0A0HYM2": 4.0 + 0.4 * np.cos(trend[:900] * 14),
        }
    ).to_csv(archive_csv, index=False)
    qqq_csv = tmp_path / "qqq_macro.csv"
    pd.DataFrame(
        {
            "trade_date": idx,
            "ticker": "QQQ",
            "close": 100.0 * np.exp(np.cumsum(0.002 + 0.01 * np.sin(trend * 16))),
            "source_name": "diagnostic_vendor",
            "fetch_timestamp": "2026-04-27 00:00",
            "price_basis": "vendor_raw_close",
        }
    ).to_csv(qqq_csv, index=False)

    def fake_fetch(series_id: str, api_key: str, start: str, end: str) -> pd.Series:
        del api_key, start, end
        return live_map[series_id]

    monkeypatch.setenv("FRED_API_KEY", "test-key")
    monkeypatch.setattr("qqq_cycle.backtest.real_replay.fetch_fred_series", fake_fetch)
    monkeypatch.setattr(
        "qqq_cycle.backtest.real_replay._load_ai_gpr",
        lambda *_: pd.Series(50.0 + 15.0 * np.sin(trend * 18), index=idx),
    )

    result = run_real_replay(
        RealReplayConfig(
            cache_root=tmp_path / "cache",
            output_dir=tmp_path / "outputs",
            qqq_price_csv=qqq_csv,
            hyoas_csv=archive_csv,
        )
    )

    manifest = json.loads(result.manifest_path.read_text())
    assert result.mode == "state_stress_only"
    assert result.weekly_replay_path is not None
    assert manifest["hyoas_source"] == "csv_override"
    assert manifest["series_coverage"]["BAMLH0A0HYM2"]["rows"] == 900


def test_real_replay_writes_hyoas_archive_manifest_and_window_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = pd.date_range("2000-01-07", "2025-12-26", freq="W-FRI")
    trend = np.linspace(0.0, 1.0, len(idx))
    live_map = {
        "DFII10": pd.Series(0.4 + 0.2 * np.sin(trend * 20), index=idx),
        "DGS2": pd.Series(2.0 + 0.5 * np.sin(trend * 12), index=idx),
        "NFCI": pd.Series(-0.25 + 0.2 * np.sin(trend * 17), index=idx),
        "VIXCLS": pd.Series(18.0 + 4.0 * np.sin(trend * 19), index=idx),
        "USEPUINDXD": pd.Series(90.0 + 20.0 * np.cos(trend * 15), index=idx),
    }
    archive_idx = idx[idx <= pd.Timestamp("2021-03-19")]
    archive_csv = tmp_path / "BAMLH0A0HYM2.csv"
    pd.DataFrame(
        {
            "DATE": archive_idx,
            "BAMLH0A0HYM2": 4.0 + 0.4 * np.cos(np.linspace(0.0, 1.0, len(archive_idx)) * 14),
        }
    ).to_csv(archive_csv, index=False)
    qqq_csv = tmp_path / "qqq_macro.csv"
    pd.DataFrame(
        {
            "trade_date": idx,
            "ticker": "QQQ",
            "close": 100.0 * np.exp(np.cumsum(0.002 + 0.01 * np.sin(trend * 16))),
            "source_name": "diagnostic_vendor",
            "fetch_timestamp": "2026-04-27 00:00",
            "price_basis": "vendor_raw_close",
        }
    ).to_csv(qqq_csv, index=False)

    monkeypatch.setenv("FRED_API_KEY", "test-key")
    monkeypatch.setattr(
        "qqq_cycle.backtest.real_replay.fetch_fred_series",
        lambda sid, *_: live_map[sid],
    )
    monkeypatch.setattr(
        "qqq_cycle.backtest.real_replay._load_ai_gpr",
        lambda *_: pd.Series(50.0 + 15.0 * np.sin(trend * 18), index=idx),
    )
    monkeypatch.setattr(
        "qqq_cycle.backtest.real_replay._utc_now_iso",
        lambda: "2026-04-27T00:00:00Z",
    )

    source_url = "https://raw.githubusercontent.com/csaladenes/eco-archive/main/BAMLH0A0HYM2.csv"
    result = run_real_replay(
        RealReplayConfig(
            cache_root=tmp_path / "cache",
            output_dir=tmp_path / "outputs",
            qqq_price_csv=qqq_csv,
            hyoas_csv=archive_csv,
            hyoas_source_url=source_url,
        )
    )

    archive_manifest = json.loads((result.output_dir / "hyoas_archive_manifest.json").read_text())
    assert archive_manifest["source_url"] == source_url
    assert archive_manifest["download_timestamp"] == "2026-04-27T00:00:00Z"
    assert archive_manifest["sha256"] == hashlib.sha256(archive_csv.read_bytes()).hexdigest()
    assert archive_manifest["min_date"] == archive_idx.min().strftime("%Y-%m-%d")
    assert archive_manifest["max_date"] == archive_idx.max().strftime("%Y-%m-%d")
    assert archive_manifest["row_count"] == len(archive_idx)
    assert archive_manifest["hyoas_source"] == "csv_override"
    assert archive_manifest["audit_grade"] == "conditional"
    assert archive_manifest["production_eligible"] is False

    coverage = pd.read_csv(result.output_dir / "event_window_coverage.csv")
    third = coverage[coverage["window"] == "2021_10_to_2022_03"].iloc[0]
    assert third["coverage_ok"] == np.False_
    assert third["window_status"] == "incomplete_due_to_hyoas_coverage"


def test_real_replay_splices_hyoas_supplement_with_boundary_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = pd.date_range("2000-01-07", "2025-12-26", freq="W-FRI")
    trend = np.linspace(0.0, 1.0, len(idx))
    live_map = {
        "DFII10": pd.Series(0.4 + 0.2 * np.sin(trend * 20), index=idx),
        "DGS2": pd.Series(2.0 + 0.5 * np.sin(trend * 12), index=idx),
        "NFCI": pd.Series(-0.25 + 0.2 * np.sin(trend * 17), index=idx),
        "VIXCLS": pd.Series(18.0 + 4.0 * np.sin(trend * 19), index=idx),
        "USEPUINDXD": pd.Series(90.0 + 20.0 * np.cos(trend * 15), index=idx),
    }
    primary_idx = idx[idx <= pd.Timestamp("2021-03-19")]
    supplement_idx = idx[
        (idx >= pd.Timestamp("2021-03-26")) & (idx <= pd.Timestamp("2022-03-31"))
    ]
    primary_csv = tmp_path / "BAMLH0A0HYM2_primary.csv"
    supplement_csv = tmp_path / "BAMLH0A0HYM2_supplement.csv"
    pd.DataFrame({"DATE": primary_idx, "BAMLH0A0HYM2": 3.0}).to_csv(
        primary_csv, index=False
    )
    pd.DataFrame({"DATE": supplement_idx, "BAMLH0A0HYM2": 3.5}).to_csv(
        supplement_csv, index=False
    )
    qqq_csv = tmp_path / "qqq_macro.csv"
    pd.DataFrame(
        {
            "trade_date": idx,
            "ticker": "QQQ",
            "close": 100.0 * np.exp(np.cumsum(0.002 + 0.01 * np.sin(trend * 16))),
            "source_name": "diagnostic_vendor",
            "fetch_timestamp": "2026-04-27 00:00",
            "price_basis": "vendor_raw_close",
        }
    ).to_csv(qqq_csv, index=False)

    monkeypatch.setenv("FRED_API_KEY", "test-key")
    monkeypatch.setattr(
        "qqq_cycle.backtest.real_replay.fetch_fred_series",
        lambda sid, *_: live_map[sid],
    )
    monkeypatch.setattr(
        "qqq_cycle.backtest.real_replay._load_ai_gpr",
        lambda *_: pd.Series(50.0 + 15.0 * np.sin(trend * 18), index=idx),
    )

    result = run_real_replay(
        RealReplayConfig(
            cache_root=tmp_path / "cache",
            output_dir=tmp_path / "outputs",
            qqq_price_csv=qqq_csv,
            hyoas_csv=primary_csv,
            hyoas_source_url="primary-url",
            hyoas_supplemental_csv=supplement_csv,
            hyoas_supplemental_source_url="supplement-url",
        )
    )

    archive_manifest = json.loads((result.output_dir / "hyoas_archive_manifest.json").read_text())
    coverage = pd.read_csv(result.output_dir / "event_window_coverage.csv")
    third = coverage[coverage["window"] == "2021_10_to_2022_03"].iloc[0]

    assert archive_manifest["max_date"] == supplement_idx.max().strftime("%Y-%m-%d")
    assert archive_manifest["supplemental_sources"][0]["source_url"] == "supplement-url"
    assert archive_manifest["boundary_checks"][0]["status"] == "ok"
    assert third["coverage_ok"] == np.True_
    assert third["window_status"] == "ok"


def test_real_replay_uses_hyoas_csv_override_for_state_stress_only_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = pd.date_range("2000-01-07", "2025-12-26", freq="W-FRI")
    trend = np.linspace(0.0, 1.0, len(idx))
    # All FRED series except BAMLH0A0HYM2 — simulates FRED API truncation for ICE BofA series
    series_map = {
        "DFII10": pd.Series(0.4 + 0.2 * np.sin(trend * 20), index=idx),
        "DGS2": pd.Series(2.0 + 0.5 * np.sin(trend * 12), index=idx),
        "NFCI": pd.Series(-0.25 + 0.2 * np.sin(trend * 17), index=idx),
        "VIXCLS": pd.Series(18.0 + 4.0 * np.sin(trend * 19), index=idx),
        "USEPUINDXD": pd.Series(90.0 + 20.0 * np.cos(trend * 15), index=idx),
    }
    ai_gpr = pd.Series(50.0 + 15.0 * np.sin(trend * 18), index=idx)

    qqq_csv = tmp_path / "qqq_macro.csv"
    qqq = 100.0 * np.exp(np.cumsum(0.002 + 0.01 * np.sin(trend * 16)))
    pd.DataFrame(
        {
            "trade_date": idx,
            "ticker": "QQQ",
            "close": qqq,
            "source_name": "diagnostic_vendor",
            "fetch_timestamp": "2026-04-27 00:00",
            "price_basis": "vendor_backward_adjusted",
        }
    ).to_csv(qqq_csv, index=False)

    hyoas_csv = tmp_path / "hyoas_long_history.csv"
    hyoas_vals = pd.Series(4.0 + 0.4 * np.cos(trend * 14), index=idx)
    pd.DataFrame({"date": idx, "BAMLH0A0HYM2": hyoas_vals.to_numpy()}).to_csv(
        hyoas_csv, index=False
    )

    def fake_fetch(series_id: str, api_key: str, start: str, end: str) -> pd.Series:
        del api_key, start, end
        if series_id not in series_map:
            raise ValueError(f"FRED API truncated: {series_id}")
        return series_map[series_id]

    monkeypatch.setenv("FRED_API_KEY", "test-key")
    monkeypatch.setattr("qqq_cycle.backtest.real_replay.fetch_fred_series", fake_fetch)
    monkeypatch.setattr("qqq_cycle.backtest.real_replay._load_ai_gpr", lambda *_: ai_gpr)

    result = run_real_replay(
        RealReplayConfig(
            cache_root=tmp_path / "cache",
            output_dir=tmp_path / "outputs",
            qqq_price_csv=qqq_csv,
            hyoas_csv=hyoas_csv,
        )
    )

    manifest = json.loads(result.manifest_path.read_text())
    assert result.mode == "state_stress_only"
    assert result.weekly_replay_path is not None
    assert manifest["hyoas_source"] == "csv_override"
    assert manifest["missing_sources"] == []
    assert "BAMLH0A0HYM2" in manifest["series_coverage"]
    cov = manifest["series_coverage"]["BAMLH0A0HYM2"]
    assert cov["rows"] > 0
    assert "min_date" in cov and "max_date" in cov


def test_real_replay_manifest_contains_series_coverage_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = pd.date_range("2000-01-07", "2025-12-26", freq="W-FRI")
    trend = np.linspace(0.0, 1.0, len(idx))
    full_series = {
        "DFII10": pd.Series(0.4 + 0.2 * np.sin(trend * 20), index=idx),
        "DGS2": pd.Series(2.0 + 0.5 * np.sin(trend * 12), index=idx),
        "BAMLH0A0HYM2": pd.Series(4.0 + 0.4 * np.cos(trend * 14), index=idx),
        "NFCI": pd.Series(-0.25 + 0.2 * np.sin(trend * 17), index=idx),
        "VIXCLS": pd.Series(18.0 + 4.0 * np.sin(trend * 19), index=idx),
        "USEPUINDXD": pd.Series(90.0 + 20.0 * np.cos(trend * 15), index=idx),
    }
    ai_gpr = pd.Series(50.0 + 15.0 * np.sin(trend * 18), index=idx)

    qqq_csv = tmp_path / "qqq_macro.csv"
    qqq = 100.0 * np.exp(np.cumsum(0.002 + 0.01 * np.sin(trend * 16)))
    pd.DataFrame(
        {
            "trade_date": idx,
            "ticker": "QQQ",
            "close": qqq,
            "source_name": "diagnostic_vendor",
            "fetch_timestamp": "2026-04-27 00:00",
            "price_basis": "vendor_backward_adjusted",
        }
    ).to_csv(qqq_csv, index=False)

    hyoas_csv = tmp_path / "hyoas_long_history.csv"
    pd.DataFrame(
        {"date": idx, "BAMLH0A0HYM2": full_series["BAMLH0A0HYM2"].to_numpy()}
    ).to_csv(hyoas_csv, index=False)

    monkeypatch.setenv("FRED_API_KEY", "test-key")
    monkeypatch.setattr(
        "qqq_cycle.backtest.real_replay.fetch_fred_series",
        lambda sid, *_: full_series[sid],
    )
    monkeypatch.setattr("qqq_cycle.backtest.real_replay._load_ai_gpr", lambda *_: ai_gpr)

    result = run_real_replay(
        RealReplayConfig(
            cache_root=tmp_path / "cache",
            output_dir=tmp_path / "outputs",
            qqq_price_csv=qqq_csv,
            hyoas_csv=hyoas_csv,
        )
    )

    manifest = json.loads(result.manifest_path.read_text())
    assert "series_coverage" in manifest
    for name in ("DFII10", "DGS2", "BAMLH0A0HYM2", "NFCI", "VIXCLS", "USEPUINDXD"):
        assert name in manifest["series_coverage"], f"{name} missing from series_coverage"
        cov = manifest["series_coverage"][name]
        assert {"rows", "min_date", "max_date"} <= cov.keys()
        assert cov["rows"] > 0
    assert "insufficient_series" in manifest
    assert isinstance(manifest["insufficient_series"], list)


def test_real_replay_falls_back_to_fred_when_hyoas_csv_load_fails(
    tmp_path: Path,
) -> None:
    config = RealReplayConfig(
        cache_root=tmp_path / "cache",
        output_dir=tmp_path / "outputs",
        fetch_fred=False,
        fetch_ai_gpr=False,
        qqq_price_csv=None,
        hyoas_csv=tmp_path / "nonexistent_hyoas.csv",
    )

    result = run_real_replay(config)

    manifest = json.loads(result.manifest_path.read_text())
    assert result.mode == "degraded"
    assert result.weekly_replay_path is None
    assert "BAMLH0A0HYM2_csv_override" in manifest["source_errors"]


def test_real_replay_auto_splice_when_fred_hyoas_is_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx_long = pd.date_range("2000-01-07", "2025-12-26", freq="W-FRI")
    trend = np.linspace(0.0, 1.0, len(idx_long))

    # All FRED series — BAMLH0A0HYM2 returns only 100 daily rows (< MIN_WARMUP_ROWS=525)
    idx_short = pd.date_range("2023-05-01", periods=100, freq="D")
    full_series: dict[str, pd.Series] = {
        "DFII10": pd.Series(0.4 + 0.2 * np.sin(trend * 20), index=idx_long),
        "DGS2": pd.Series(2.0 + 0.5 * np.sin(trend * 12), index=idx_long),
        "BAMLH0A0HYM2": pd.Series(4.0 + 0.1 * np.arange(100), index=idx_short),
        "NFCI": pd.Series(-0.25 + 0.2 * np.sin(trend * 17), index=idx_long),
        "VIXCLS": pd.Series(18.0 + 4.0 * np.sin(trend * 19), index=idx_long),
        "USEPUINDXD": pd.Series(90.0 + 20.0 * np.cos(trend * 15), index=idx_long),
        # BAA10Y fetched internally by the splice builder
        "BAA10Y": pd.Series(1.5 + 0.3 * np.sin(trend * 11), index=idx_long),
    }
    ai_gpr = pd.Series(50.0 + 15.0 * np.sin(trend * 18), index=idx_long)

    qqq_csv = tmp_path / "qqq_macro.csv"
    qqq = 100.0 * np.exp(np.cumsum(0.002 + 0.01 * np.sin(trend * 16)))
    pd.DataFrame(
        {
            "trade_date": idx_long,
            "ticker": "QQQ",
            "close": qqq,
            "source_name": "diagnostic_vendor",
            "fetch_timestamp": "2026-04-27 00:00",
            "price_basis": "vendor_backward_adjusted",
        }
    ).to_csv(qqq_csv, index=False)

    def fake_fetch(series_id: str, api_key: str, start: str, end: str) -> pd.Series:
        del api_key, start, end
        return full_series[series_id]

    monkeypatch.setenv("FRED_API_KEY", "test-key")
    monkeypatch.setattr("qqq_cycle.backtest.real_replay.fetch_fred_series", fake_fetch)
    monkeypatch.setattr("qqq_cycle.backtest.real_replay._load_ai_gpr", lambda *_: ai_gpr)

    result = run_real_replay(
        RealReplayConfig(
            cache_root=tmp_path / "cache",
            output_dir=tmp_path / "outputs",
            qqq_price_csv=qqq_csv,
        )
    )

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["hyoas_source"] == "baa10y_splice"
    assert manifest["hyoas_splice_method"] is not None
    assert "OLS(BAA10Y" in manifest["hyoas_splice_method"]
    assert "BAMLH0A0HYM2" in manifest["active_sources"]
    # The splice should produce enough history to pass the warmup gate
    assert result.mode == "state_stress_only"
    assert result.weekly_replay_path is not None
