"""Cache-backed real/partial state replay loader with explicit degradation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlencode
from urllib.request import urlopen

import pandas as pd
from dotenv import load_dotenv

from qqq_cycle.backtest.diagnostics import build_replay_bundle, write_replay_outputs
from qqq_cycle.backtest.oos_eval import (
    summarize_numerical_health,
    write_health_summary,
    write_tail_diagnostics,
)

DEFAULT_FRED_SERIES = (
    "DFII10",
    "DGS2",
    "BAMLH0A0HYM2",
    "NFCI",
    "VIXCLS",
    "USEPUINDXD",
)

AI_GPR_OFFICIAL_PAGE = "https://www.matteoiacoviello.com/ai_gpr.html"
REQUIRED_STATE_COLUMNS = [
    "DFII10",
    "DGS2",
    "BAMLH0A0HYM2",
    "NFCI",
    "VIXCLS",
    "USEPUINDXD",
    "AI_GPR",
    "QQQ",
]


@dataclass(frozen=True)
class RealReplayConfig:
    """Configuration for cache-backed real replay generation."""

    cache_root: Path = Path("cache/real_replay")
    output_dir: Path = Path("outputs/replay/real")
    start: str = "2000-01-01"
    end: str = "2025-12-31"
    fred_series: tuple[str, ...] = DEFAULT_FRED_SERIES
    fetch_fred: bool = True
    fetch_ai_gpr: bool = True
    qqq_price_csv: Path | None = None
    ai_gpr_url: str = AI_GPR_OFFICIAL_PAGE


@dataclass(frozen=True)
class RealReplayResult:
    """Paths and mode emitted by real replay runner."""

    mode: str
    output_dir: Path
    manifest_path: Path
    weekly_replay_path: Path | None


def _ensure_cache_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "raw": root / "raw",
        "staging": root / "staging",
        "manifests": root / "manifests",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def fetch_fred_series(series_id: str, api_key: str, start: str, end: str) -> pd.Series:
    """Fetch one FRED series as daily observations from the official API."""

    params = urlencode(
        {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
        }
    )
    url = f"https://api.stlouisfed.org/fred/series/observations?{params}"
    with urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("observations", [])
    dates = []
    values = []
    for row in rows:
        value = row.get("value")
        if value in (None, "."):
            continue
        dates.append(pd.Timestamp(row["date"]))
        values.append(float(value))
    return pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id).sort_index()


def _resolve_ai_gpr_csv_url(url: str) -> str:
    if url.endswith(".csv"):
        return url
    with urlopen(url, timeout=30) as response:
        html = response.read().decode("utf-8")
    marker = "ai_gpr_files/ai_gpr_data_daily.csv"
    if marker not in html:
        raise ValueError("official AI-GPR page did not contain daily CSV link")
    return urljoin(url, marker)


def _load_ai_gpr(url: str, start: str, end: str) -> pd.Series:
    csv_url = _resolve_ai_gpr_csv_url(url)
    raw = pd.read_csv(csv_url)
    date_col = next((c for c in raw.columns if c.lower() in {"date", "day"}), None)
    value_col = next(
        (
            c
            for c in raw.columns
            if c.lower() in {"ai_gpr", "aigpr", "ai-gpr", "index", "ai_gpr_index", "gpr_ai"}
        ),
        None,
    )
    if date_col is None or value_col is None:
        raise ValueError("AI-GPR CSV did not expose recognizable date/value columns")
    series = pd.Series(
        pd.to_numeric(raw[value_col], errors="coerce").to_numpy(),
        index=pd.to_datetime(raw[date_col]),
        name="AI_GPR",
    ).dropna()
    return series.loc[(series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))]


def _load_qcc_csv(path: Path, start: str, end: str) -> pd.Series:
    raw = pd.read_csv(path)
    date_col = next((c for c in raw.columns if c.lower() in {"date", "trade_date"}), None)
    value_col = next((c for c in raw.columns if c.lower() in {"close", "raw_close"}), None)
    if date_col is None or value_col is None:
        raise ValueError("QQQ CSV must include date/trade_date and close/raw_close columns")
    series = pd.Series(
        pd.to_numeric(raw[value_col], errors="coerce").to_numpy(),
        index=pd.to_datetime(raw[date_col]),
        name="QQQ",
    ).dropna()
    return series.loc[(series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))]


def _weekly(series: pd.Series) -> pd.Series:
    return series.sort_index().resample("W-FRI").last()


def _stage_weekly_inputs(series_map: dict[str, pd.Series], staging_dir: Path) -> pd.DataFrame:
    frame = pd.concat({name: _weekly(series) for name, series in series_map.items()}, axis=1)
    frame.to_csv(staging_dir / "weekly_inputs.csv", index_label="week_end")
    return frame


def _missing(required: Iterable[str], series_map: dict[str, pd.Series]) -> list[str]:
    return [name for name in required if name not in series_map or series_map[name].dropna().empty]


def run_real_replay(config: RealReplayConfig) -> RealReplayResult:
    """Run official-source replay when possible, otherwise degrade explicitly."""

    load_dotenv(".env")
    dirs = _ensure_cache_dirs(config.cache_root)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    missing_sources: list[str] = []
    source_errors: dict[str, str] = {}
    series_map: dict[str, pd.Series] = {}

    fred_key = os.getenv("FRED_API_KEY", "")
    if config.fetch_fred and fred_key:
        for series_id in config.fred_series:
            try:
                series = fetch_fred_series(series_id, fred_key, config.start, config.end)
                series.to_csv(dirs["raw"] / f"fred_{series_id}.csv", index_label="date")
                series_map[series_id] = series
            except Exception as exc:  # noqa: BLE001 - stored as degradation metadata
                missing_sources.append(series_id)
                source_errors[series_id] = str(exc)
    else:
        missing_sources.extend(config.fred_series)
        if not fred_key:
            source_errors["FRED_API_KEY"] = "missing from .env"

    if config.fetch_ai_gpr:
        try:
            ai_gpr = _load_ai_gpr(config.ai_gpr_url, config.start, config.end)
            ai_gpr.to_csv(dirs["raw"] / "ai_gpr_data_daily.csv", index_label="date")
            series_map["AI_GPR"] = ai_gpr
        except Exception as exc:  # noqa: BLE001 - official source may be unavailable
            missing_sources.append("AI_GPR")
            source_errors["AI_GPR"] = str(exc)
    else:
        missing_sources.append("AI_GPR")
        source_errors["AI_GPR"] = "fetch disabled"

    if config.qqq_price_csv is not None:
        try:
            qqq = _load_qcc_csv(config.qqq_price_csv, config.start, config.end)
            qqq.to_csv(dirs["raw"] / "qqq_raw_close.csv", index_label="date")
            series_map["QQQ"] = qqq
        except Exception as exc:  # noqa: BLE001
            missing_sources.append("QQQ")
            source_errors["QQQ"] = str(exc)
    else:
        missing_sources.append("QQQ")
        source_errors["QQQ"] = "no official PIT/raw close source configured"

    weekly = _stage_weekly_inputs(series_map, dirs["staging"]) if series_map else pd.DataFrame()
    missing_sources = sorted(set(missing_sources + _missing(REQUIRED_STATE_COLUMNS, series_map)))
    mode = "full" if not missing_sources else "degraded"
    weekly_replay_path: Path | None = None
    if mode == "full":
        bundle = build_replay_bundle(weekly[REQUIRED_STATE_COLUMNS].dropna(how="all"))
        write_replay_outputs(bundle, config.output_dir)
        summary = summarize_numerical_health(bundle.weekly)
        write_health_summary(summary, config.output_dir)
        write_tail_diagnostics(bundle.weekly, config.output_dir)
        weekly_replay_path = config.output_dir / "weekly_replay.csv"
    else:
        _write_json(config.output_dir / "missing_sources.json", {"missing_sources": missing_sources, "source_errors": source_errors})

    manifest = {
        "mode": mode,
        "cache_layout": {
            "raw": str(dirs["raw"]),
            "staging": str(dirs["staging"]),
            "manifests": str(dirs["manifests"]),
        },
        "missing_sources": missing_sources,
        "source_errors": source_errors,
        "weekly_rows_staged": int(len(weekly)),
        "outputs": str(config.output_dir),
    }
    manifest_path = dirs["manifests"] / "real_replay_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(config.output_dir / "metadata.json", manifest)
    return RealReplayResult(mode=mode, output_dir=config.output_dir, manifest_path=manifest_path, weekly_replay_path=weekly_replay_path)
