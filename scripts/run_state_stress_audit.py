from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qqq_cycle.backtest.state_stress_audit import (
    build_warmup_dependency_map,
    build_replay_with_hyoas_source,
    explain_warmup_boundary,
    freeze_replay_baseline,
    sha256_file,
    summarize_source_sensitivity,
    write_behavior_audits,
    write_source_sensitivity_report,
    write_warmup_explanation,
)


def _commit_hash() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _load_hyoas_series(path: Path) -> pd.Series:
    raw = pd.read_csv(path)
    lower = {str(col).strip().lower(): col for col in raw.columns}
    date_col = next(
        (
            lower[name]
            for name in ("date", "day", "week_end", "observation_date", "yyyymmdd", "index")
            if name in lower
        ),
        raw.columns[0],
    )
    value_col = next(
        (
            lower[name]
            for name in ("bamlh0a0hym2", "hyoas", "hy_oas", "hy_spread", "value")
            if name in lower and lower[name] != date_col
        ),
        next(col for col in raw.columns if col != date_col),
    )
    if str(date_col).lower() == "yyyymmdd":
        dates = pd.to_datetime(raw[date_col].astype(str), format="%Y%m%d", errors="coerce")
    else:
        dates = pd.to_datetime(raw[date_col], errors="coerce")
    series = pd.Series(
        pd.to_numeric(raw[value_col], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(dates),
        name="BAMLH0A0HYM2",
    ).dropna()
    return series.sort_index()


def _load_weekly_inputs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["week_end"] = pd.to_datetime(frame["week_end"])
    return frame


def _raw_first_dates(raw_dir: Path) -> dict[str, str | None]:
    paths = {
        "DFII10": raw_dir / "fred_DFII10.csv",
        "DGS2": raw_dir / "fred_DGS2.csv",
        "BAMLH0A0HYM2": raw_dir / "fred_BAMLH0A0HYM2.csv",
        "NFCI": raw_dir / "fred_NFCI.csv",
        "VIXCLS": raw_dir / "fred_VIXCLS.csv",
        "USEPUINDXD": raw_dir / "fred_USEPUINDXD.csv",
        "AI_GPR": raw_dir / "ai_gpr_data_daily.csv",
        "QQQ": raw_dir / "qqq_macro_close.csv",
    }
    out: dict[str, str | None] = {}
    for name, path in paths.items():
        if not path.exists():
            out[name] = None
            continue
        series = _load_hyoas_series(path) if name == "BAMLH0A0HYM2" else None
        if series is None:
            raw = pd.read_csv(path)
            date_col = raw.columns[0]
            value_col = next(col for col in raw.columns if col != date_col)
            series = pd.Series(
                pd.to_numeric(raw[value_col], errors="coerce").to_numpy(),
                index=pd.to_datetime(raw[date_col], errors="coerce"),
            ).dropna()
        out[name] = None if series.empty else series.index.min().strftime("%Y-%m-%d")
    return out


def main() -> None:
    replay_dir = ROOT / "outputs" / "replay" / "real"
    audit_dir = ROOT / "outputs" / "audit" / "state_stress_replay"
    cache_raw = ROOT / "cache" / "real_replay" / "raw"
    weekly_inputs_path = ROOT / "cache" / "real_replay" / "staging" / "weekly_inputs.csv"

    commit = _commit_hash()
    freeze_replay_baseline(replay_dir=replay_dir, audit_dir=audit_dir, commit_hash=commit)

    replay = pd.read_csv(replay_dir / "weekly_replay.csv")
    weekly_inputs = _load_weekly_inputs(weekly_inputs_path)
    warmup_map = build_warmup_dependency_map(
        weekly_inputs,
        replay,
        raw_first_dates=_raw_first_dates(cache_raw),
    )
    warmup_explanation = explain_warmup_boundary(
        warmup_map,
        window_name="2008_09_to_2009_06",
        start="2008-09-01",
        end="2009-06-30",
    )
    write_warmup_explanation(warmup_map, warmup_explanation, audit_dir)
    write_behavior_audits(
        replay,
        audit_dir,
        warmup_explanations={"2008_09_to_2009_06": warmup_explanation},
    )

    hyoas_sources: dict[str, Path] = {
        "eco_archive_only": ROOT / "scratch" / "hyoas_csaladenes.csv",
        "eco_archive_plus_equibles": cache_raw / "fred_BAMLH0A0HYM2.csv",
    }
    splice = cache_raw / "hyoas_baa10y_splice.csv"
    if splice.exists():
        hyoas_sources["baa10y_splice_experimental"] = splice

    sensitivity_replays: dict[str, pd.DataFrame] = {}
    sensitivity_paths: dict[str, str] = {}
    sensitivity_dir = audit_dir / "source_replays"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    for source_name, source_path in hyoas_sources.items():
        if not source_path.exists():
            continue
        hyoas = _load_hyoas_series(source_path)
        weekly = build_replay_with_hyoas_source(weekly_inputs, hyoas)
        sensitivity_replays[source_name] = weekly
        out_path = sensitivity_dir / f"weekly_replay_{source_name}.csv"
        weekly.to_csv(out_path, index=False)
        sensitivity_paths[source_name] = str(out_path)

    sensitivity = summarize_source_sensitivity(
        sensitivity_replays,
        reference_source="eco_archive_plus_equibles",
    )
    write_source_sensitivity_report(sensitivity, audit_dir)

    manifest_path = audit_dir / "audit_baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in sorted(audit_dir.glob("*.csv")) + sorted(audit_dir.glob("*.md")) + sorted(audit_dir.glob("*.json")):
        manifest["file_hashes"][str(path)] = sha256_file(path)
    for source_name, path in sensitivity_paths.items():
        manifest["file_hashes"][path] = sha256_file(path)
    manifest["source_sensitivity_replays"] = sensitivity_paths
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"audit_dir={audit_dir}")
    print(f"manifest={manifest_path}")
    print(f"sources={','.join(sensitivity_replays)}")


if __name__ == "__main__":
    main()
