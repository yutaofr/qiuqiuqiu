"""Cache-backed real/partial state replay loader with explicit degradation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from qqq_cycle.backtest.diagnostics import build_replay_bundle, write_replay_outputs
from qqq_cycle.backtest.oos_eval import (
    summarize_numerical_health,
    write_health_summary,
    write_tail_diagnostics,
)
from qqq_cycle.data_contracts.macro_prices import (
    ALLOWED_REPLAY_SCOPE,
    CsvMacroMarketPriceStore,
    PriceBasis,
)

DEFAULT_FRED_SERIES = (
    "DFII10",
    "DGS2",
    "BAMLH0A0HYM2",
    "NFCI",
    "VIXCLS",
    "USEPUINDXD",
)

# 265 theta minimum + 260 dual-memory warmup buffer; advisory threshold for _insufficient()
MIN_WARMUP_ROWS = 1325

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
    # CSV override for BAMLH0A0HYM2 when FRED API returns truncated ICE BofA history
    # (~3 years from April 2023). Must have a date column and one numeric value column;
    # column names are detected flexibly. Recommended span >=10 years (>=525 daily rows).
    # When None and FRED history is insufficient, an auto-splice is built from BAA10Y.
    hyoas_csv: Path | None = None
    hyoas_source_url: str | None = None
    hyoas_supplemental_csv: Path | None = None
    hyoas_supplemental_source_url: str | None = None
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


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _load_qqq_macro_csv(
    path: Path, start: str, end: str
) -> tuple[pd.Series, PriceBasis, tuple[str, ...]]:
    store = CsvMacroMarketPriceStore(path, replay_scope=ALLOWED_REPLAY_SCOPE)
    return (
        store.to_series("QQQ", pd.Timestamp(start), pd.Timestamp(end)),
        store.price_basis,
        store.source_names,
    )


def _load_hyoas_csv(path: Path, start: str, end: str) -> pd.Series:
    """Load a long-history BAMLH0A0HYM2 CSV when FRED API history is truncated."""
    raw = pd.read_csv(path)
    lower = [c.lower().strip() for c in raw.columns]
    date_col = next(
        (raw.columns[i] for i, n in enumerate(lower) if n in {"date", "day", "week_end", "index"}),
        raw.columns[0],
    )
    value_col = next(
        (
            raw.columns[i]
            for i, n in enumerate(lower)
            if raw.columns[i] != date_col
            and n in {"bamlh0a0hym2", "hyoas", "hy_oas", "hy_spread", "value"}
        ),
        next((c for c in raw.columns if c != date_col), None),
    )
    if value_col is None:
        raise ValueError("HYOAS CSV has no usable value column")
    series = pd.Series(
        pd.to_numeric(raw[value_col], errors="coerce").to_numpy(),
        index=pd.to_datetime(raw[date_col]),
        name="BAMLH0A0HYM2",
    ).dropna()
    series = series.loc[
        (series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))
    ]
    if series.empty:
        raise ValueError(f"HYOAS CSV has no rows in [{start}, {end}]")
    return series.sort_index()


def _archive_manifest(
    *,
    path: Path,
    source_url: str | None,
    series: pd.Series,
    hyoas_source: str,
    supplemental_sources: list[dict[str, object]] | None = None,
    boundary_checks: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    finite = series.dropna()
    return {
        "source_url": source_url or str(path),
        "download_timestamp": _utc_now_iso(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "min_date": finite.index.min().strftime("%Y-%m-%d"),
        "max_date": finite.index.max().strftime("%Y-%m-%d"),
        "row_count": int(len(finite)),
        "hyoas_source": hyoas_source,
        "audit_grade": "conditional",
        "production_eligible": False,
        "supplemental_sources": supplemental_sources or [],
        "boundary_checks": boundary_checks or [],
    }


def _splice_hyoas_supplement(
    primary: pd.Series,
    supplement: pd.Series,
    *,
    tolerance: float = 1e-8,
) -> tuple[pd.Series, list[dict[str, object]]]:
    overlap = primary.index.intersection(supplement.index)
    checks: list[dict[str, object]] = []
    if len(overlap) > 0:
        diff = (primary.loc[overlap] - supplement.loc[overlap]).abs()
        max_abs_diff = float(diff.max())
        status = "ok" if max_abs_diff <= tolerance else "conflict"
        checks.append(
            {
                "overlap_rows": int(len(overlap)),
                "max_abs_diff": max_abs_diff,
                "tolerance": tolerance,
                "status": status,
            }
        )
        if status != "ok":
            raise ValueError(
                f"HYOAS supplemental overlap conflict: max_abs_diff={max_abs_diff}"
            )
    else:
        checks.append(
            {
                "overlap_rows": 0,
                "max_abs_diff": None,
                "tolerance": tolerance,
                "status": "ok",
            }
        )
    combined = pd.concat([primary, supplement[~supplement.index.isin(primary.index)]])
    return combined.sort_index(), checks


def _build_hyoas_splice(
    fred_key: str,
    start: str,
    end: str,
    series_map: dict[str, pd.Series],
    raw_dir: Path,
) -> tuple[pd.Series, str]:
    """Regression-splice for BAMLH0A0HYM2 when FRED API history is truncated.

    Calibrates OLS(BAA10Y, log(VIX), NFCI) on the available BAMLH0A0HYM2 overlap,
    predicts the pre-overlap period at daily frequency, then concatenates with the
    actual FRED observations. BAA10Y is a Fed series with free full history to 1962.
    """
    baa10y = fetch_fred_series("BAA10Y", fred_key, start, end)
    baa10y.to_csv(raw_dir / "fred_BAA10Y.csv", index_label="date")

    hyoas_actual = series_map.get("BAMLH0A0HYM2", pd.Series([], dtype=float, name="BAMLH0A0HYM2"))
    vix = series_map.get("VIXCLS")
    nfci = series_map.get("NFCI")

    def w(s: pd.Series) -> pd.Series:
        return s.sort_index().resample("W-FRI").last()

    pieces: dict[str, pd.Series] = {"baa": w(baa10y)}
    if vix is not None:
        pieces["lvix"] = np.log(w(vix).clip(lower=1.0))
    if nfci is not None:
        pieces["nfci"] = w(nfci)

    overlap = pd.concat({**pieces, "hy": w(hyoas_actual)}, axis=1, sort=True).dropna()
    # With fewer than 2*(n_predictors+1) overlap rows the multivariate fit is unstable;
    # fall back to BAA10Y-only regression which needs only 2 rows.
    pred_cols = list(pieces.keys())
    n_params = len(pred_cols) + 1
    if len(overlap) < 2 * n_params:
        pred_cols = ["baa"]
    X = np.column_stack([overlap[c].to_numpy(dtype=float) for c in pred_cols] + [np.ones(len(overlap))])
    y = overlap["hy"].to_numpy(dtype=float)
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred_in_sample = X @ coef
    ss_res = float(np.sum((y - pred_in_sample) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    daily_pieces: dict[str, pd.Series] = {"baa": baa10y}
    if vix is not None:
        daily_pieces["lvix"] = np.log(vix.clip(lower=1.0))
    if nfci is not None:
        daily_pieces["nfci"] = nfci

    full = pd.concat(daily_pieces, axis=1, sort=True).dropna()
    X_full = np.column_stack([full[c].to_numpy(dtype=float) for c in pred_cols] + [np.ones(len(full))])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        raw_pred = X_full @ coef
    # Guard against overflow/NaN from an unstable regression; cap to a plausible OAS range.
    raw_pred = np.where(np.isfinite(raw_pred), raw_pred, np.nan)
    raw_pred = np.clip(raw_pred, 0.5, 50.0)
    predicted = pd.Series(raw_pred.astype(float), index=full.index, name="BAMLH0A0HYM2")

    if not hyoas_actual.empty:
        cutoff = hyoas_actual.index.min()
        pre = predicted[predicted.index < cutoff]
        splice = pd.concat([pre, hyoas_actual]).sort_index()
    else:
        splice = predicted

    extra = "," + ",".join(p for p in pred_cols if p != "baa") if len(pred_cols) > 1 else ""
    method = f"OLS(BAA10Y{extra}) R²={r2:.3f} on {len(overlap)}w overlap"
    return splice, method


def _weekly(series: pd.Series) -> pd.Series:
    return series.sort_index().resample("W-FRI").last().ffill()


def _stage_weekly_inputs(series_map: dict[str, pd.Series], staging_dir: Path) -> pd.DataFrame:
    frame = pd.concat({name: _weekly(series) for name, series in series_map.items()}, axis=1)
    frame = frame.ffill()
    frame.to_csv(staging_dir / "weekly_inputs.csv", index_label="week_end")
    return frame


def _missing(required: Iterable[str], series_map: dict[str, pd.Series]) -> list[str]:
    return [name for name in required if name not in series_map or series_map[name].dropna().empty]


def _insufficient(
    required: Iterable[str],
    series_map: dict[str, pd.Series],
    min_rows: int = MIN_WARMUP_ROWS,
) -> list[str]:
    """Advisory: series present but below min_rows non-NaN daily rows."""
    return [
        name
        for name in required
        if name in series_map and 0 < series_map[name].dropna().shape[0] < min_rows
    ]


def _window_coverage(
    output_dir: Path,
    hyoas_max_date: str | None,
) -> pd.DataFrame:
    from qqq_cycle.backtest.diagnostics import EVENT_WINDOWS

    rows: list[dict[str, object]] = []
    hyoas_max = pd.Timestamp(hyoas_max_date) if hyoas_max_date else None
    for name, (_, end) in EVENT_WINDOWS.items():
        path = output_dir / f"event_{name}.csv"
        frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
        rows_total = int(len(frame))
        rows_finite_h = int(pd.to_numeric(frame.get("H_t"), errors="coerce").notna().sum()) if rows_total else 0
        rows_finite_s = int(pd.to_numeric(frame.get("s_t"), errors="coerce").notna().sum()) if rows_total else 0
        coverage_ok = rows_total > 0 and rows_finite_h / rows_total >= 0.95
        window_status = "ok" if coverage_ok else "incomplete"
        last_week_end = (
            pd.to_datetime(frame["week_end"]).max()
            if rows_total and "week_end" in frame
            else pd.Timestamp(end)
        )
        if hyoas_max is not None and hyoas_max < last_week_end:
            window_status = "incomplete_due_to_hyoas_coverage"
            coverage_ok = False
        rows.append(
            {
                "window": name,
                "rows_total": rows_total,
                "rows_finite_H_t": rows_finite_h,
                "rows_finite_s_t": rows_finite_s,
                "coverage_ok": bool(coverage_ok),
                "window_status": window_status,
            }
        )
    return pd.DataFrame(rows)


def run_real_replay(config: RealReplayConfig) -> RealReplayResult:
    """Run official-source replay when possible, otherwise degrade explicitly."""

    load_dotenv(".env")
    dirs = _ensure_cache_dirs(config.cache_root)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    missing_sources: list[str] = []
    source_errors: dict[str, str] = {}
    series_map: dict[str, pd.Series] = {}
    qqq_price_basis: str | None = None
    qqq_source_names: tuple[str, ...] = ()
    hyoas_source: str = "fred_api"
    hyoas_splice_method: str | None = None
    hyoas_archive_manifest: dict[str, object] | None = None
    hyoas_supplemental_sources: list[dict[str, object]] = []
    hyoas_boundary_checks: list[dict[str, object]] = []

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

    # Explicit CSV override: wins over any FRED result when provided and loads successfully.
    if config.hyoas_csv is not None:
        try:
            hyoas = _load_hyoas_csv(config.hyoas_csv, config.start, config.end)
            if config.hyoas_supplemental_csv is not None:
                supplemental = _load_hyoas_csv(
                    config.hyoas_supplemental_csv, config.start, config.end
                )
                hyoas, hyoas_boundary_checks = _splice_hyoas_supplement(
                    hyoas, supplemental
                )
                supplemental_finite = supplemental.dropna()
                hyoas_supplemental_sources.append(
                    {
                        "source_url": config.hyoas_supplemental_source_url
                        or str(config.hyoas_supplemental_csv),
                        "sha256": hashlib.sha256(
                            Path(config.hyoas_supplemental_csv).read_bytes()
                        ).hexdigest(),
                        "min_date": supplemental_finite.index.min().strftime("%Y-%m-%d"),
                        "max_date": supplemental_finite.index.max().strftime("%Y-%m-%d"),
                        "row_count": int(len(supplemental_finite)),
                    }
                )
            hyoas.to_csv(dirs["raw"] / "fred_BAMLH0A0HYM2.csv", index_label="date")
            series_map["BAMLH0A0HYM2"] = hyoas
            hyoas_source = "csv_override"
            hyoas_archive_manifest = _archive_manifest(
                path=Path(config.hyoas_csv),
                source_url=config.hyoas_source_url,
                series=hyoas,
                hyoas_source=hyoas_source,
                supplemental_sources=hyoas_supplemental_sources,
                boundary_checks=hyoas_boundary_checks,
            )
            missing_sources = [s for s in missing_sources if s != "BAMLH0A0HYM2"]
            source_errors.pop("BAMLH0A0HYM2", None)
        except Exception as exc:  # noqa: BLE001 - CSV load failure is non-fatal
            source_errors["BAMLH0A0HYM2_csv_override"] = str(exc)

    # Auto-splice: if BAMLH0A0HYM2 is still insufficient (FRED API licensing truncation)
    # and no explicit CSV was provided, build a regression splice from BAA10Y + VIX + NFCI.
    _hyoas_rows = series_map.get("BAMLH0A0HYM2", pd.Series([], dtype=float)).dropna().shape[0]
    if hyoas_source == "fred_api" and _hyoas_rows < MIN_WARMUP_ROWS and fred_key:
        try:
            splice, method = _build_hyoas_splice(
                fred_key, config.start, config.end, series_map, dirs["raw"]
            )
            splice.to_csv(dirs["raw"] / "hyoas_baa10y_splice.csv", index_label="date")
            series_map["BAMLH0A0HYM2"] = splice
            hyoas_source = "baa10y_splice"
            hyoas_splice_method = method
            missing_sources = [s for s in missing_sources if s != "BAMLH0A0HYM2"]
            source_errors.pop("BAMLH0A0HYM2", None)
        except Exception as exc:  # noqa: BLE001 - splice failure falls through to degraded
            source_errors["BAMLH0A0HYM2_splice"] = str(exc)

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
            qqq, price_basis, source_names = _load_qqq_macro_csv(
                config.qqq_price_csv, config.start, config.end
            )
            qqq_price_basis = price_basis.value
            qqq_source_names = source_names
            qqq.to_csv(dirs["raw"] / "qqq_macro_close.csv", index_label="date")
            series_map["QQQ"] = qqq
        except Exception as exc:  # noqa: BLE001
            missing_sources.append("QQQ")
            source_errors["QQQ"] = str(exc)
    else:
        missing_sources.append("QQQ")
        source_errors["QQQ"] = "no macro market price source configured"

    weekly = _stage_weekly_inputs(series_map, dirs["staging"]) if series_map else pd.DataFrame()
    missing_sources = sorted(set(missing_sources + _missing(REQUIRED_STATE_COLUMNS, series_map)))
    active_sources = [
        name
        for name in REQUIRED_STATE_COLUMNS
        if name in series_map and name not in missing_sources
    ]
    mode = ALLOWED_REPLAY_SCOPE if not missing_sources else "degraded"
    weekly_replay_path: Path | None = None
    if mode == ALLOWED_REPLAY_SCOPE:
        try:
            bundle = build_replay_bundle(weekly[REQUIRED_STATE_COLUMNS].dropna(how="all"))
            write_replay_outputs(bundle, config.output_dir)
            summary = summarize_numerical_health(bundle.weekly)
            write_health_summary(summary, config.output_dir)
            write_tail_diagnostics(bundle.weekly, config.output_dir)
            weekly_replay_path = config.output_dir / "weekly_replay.csv"
        except RuntimeError as exc:
            mode = "degraded"
            missing_sources = sorted(set(missing_sources + ["REPLAY_WARMUP"]))
            source_errors["REPLAY_WARMUP"] = str(exc)

    if mode == "degraded":
        _write_json(
            config.output_dir / "missing_sources.json",
            {"missing_sources": missing_sources, "source_errors": source_errors},
        )
    else:
        stale_missing = config.output_dir / "missing_sources.json"
        if stale_missing.exists():
            stale_missing.unlink()

    series_coverage = {
        name: {
            "rows": int(series_map[name].dropna().shape[0]),
            "min_date": series_map[name].dropna().index.min().strftime("%Y-%m-%d"),
            "max_date": series_map[name].dropna().index.max().strftime("%Y-%m-%d"),
        }
        for name in REQUIRED_STATE_COLUMNS
        if name in series_map and not series_map[name].dropna().empty
    }
    insufficient_series = _insufficient(REQUIRED_STATE_COLUMNS, series_map)
    if hyoas_archive_manifest is not None:
        _write_json(config.output_dir / "hyoas_archive_manifest.json", hyoas_archive_manifest)
    hyoas_max_date = series_coverage.get("BAMLH0A0HYM2", {}).get("max_date")
    if weekly_replay_path is not None:
        coverage = _window_coverage(config.output_dir, hyoas_max_date)
        coverage.to_csv(config.output_dir / "event_window_coverage.csv", index=False)

    manifest = {
        "mode": mode,
        "replay_scope": ALLOWED_REPLAY_SCOPE,
        "active_sources": active_sources,
        "missing_sources": missing_sources,
        "data_integrity": {
            "required_columns": REQUIRED_STATE_COLUMNS,
            "weekly_rows_staged": int(len(weekly)),
            "complete_state_inputs": not missing_sources,
            "source_errors": source_errors,
        },
        "micro_scope": "disabled_no_pit_micro_contract",
        "risk_scope": "disabled_no_production_rho_t",
        "price_basis": qqq_price_basis,
        "qqq_source_names": list(qqq_source_names),
        "hyoas_source": hyoas_source,
        "hyoas_archive_manifest": hyoas_archive_manifest,
        "hyoas_splice_method": hyoas_splice_method,
        "series_coverage": series_coverage,
        "insufficient_series": insufficient_series,
        "cache_layout": {
            "raw": str(dirs["raw"]),
            "staging": str(dirs["staging"]),
            "manifests": str(dirs["manifests"]),
        },
        "source_errors": source_errors,
        "weekly_rows_staged": int(len(weekly)),
        "outputs": str(config.output_dir),
    }
    manifest_path = dirs["manifests"] / "real_replay_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(config.output_dir / "metadata.json", manifest)
    return RealReplayResult(
        mode=mode,
        output_dir=config.output_dir,
        manifest_path=manifest_path,
        weekly_replay_path=weekly_replay_path,
    )
