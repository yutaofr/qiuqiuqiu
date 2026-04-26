import json
from pathlib import Path

from qqq_cycle.backtest.real_replay import RealReplayConfig, run_real_replay


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
