from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_phase15_reporting_and_cli(tmp_path: Path) -> None:
    phase14_path = tmp_path / "phase14_snapshot.json"
    phase14_path.write_text(
        json.dumps(
            {
                "week_end": "2026-04-24",
                "mode": "degraded",
                "backfill_mode": "degraded_backfill",
                "strict_gate_passed": False,
                "execution_permitted": False,
                "h_t": None,
                "rho_t": None,
                "k_hat_t": None,
                "s_t": None,
                "source_hash": "phase14hash",
            }
        ),
        encoding="utf-8",
    )
    portfolio_path = tmp_path / "current_positions.csv"
    portfolio_path.write_text(
        "account_id,week_end,symbol,quantity,market_price,market_value,weight,cash,source,paper_only,broker_submission_allowed\n"
        "acct,2026-04-24,QQQ,1.2,500.0,600.0,0.6,0.0,test,true,false\n"
        "acct,2026-04-24,BIL,4.0,100.0,400.0,0.4,0.0,test,true,false\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "phase15"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_phase15_sandbox.py",
            "--week-end",
            "2026-04-24",
            "--phase14-snapshot",
            str(phase14_path),
            "--portfolio-snapshot",
            str(portfolio_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=Path.cwd(),
    )

    target_path = output_dir / "target_weights_2026-04-24.json"
    delta_path = output_dir / "portfolio_delta_2026-04-24.json"
    orders_path = output_dir / "hypothetical_orders_2026-04-24.csv"
    report_path = output_dir / "execution_sandbox_report_2026-04-24.md"
    summary_path = output_dir / "execution_sandbox_summary_latest.json"

    assert target_path.exists()
    assert delta_path.exists()
    assert orders_path.exists()
    assert report_path.exists()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")

    assert summary["phase14_snapshot_hash"] == "phase14hash"
    assert summary["orders_count"] == 0
    assert summary["paper_only"] is True
    assert summary["broker_submission_allowed"] is False
    assert summary["known_limitations"] == ["discrete_bucket_turnover_risk"]
    assert "当前 Target Weights 采用阶梯式离散映射" in report


def test_summary_uses_signal_gate_flags_not_reason_inference(tmp_path: Path) -> None:
    phase14_path = tmp_path / "phase14_snapshot.json"
    phase14_path.write_text(
        json.dumps(
            {
                "week_end": "2026-04-24",
                "mode": "strict",
                "backfill_mode": "strict_recovery",
                "strict_gate_passed": True,
                "execution_permitted": True,
                "h_t": 0.2,
                "rho_t": 0.35,
                "k_hat_t": 2,
                "s_t": 0.1,
                "source_hash": "phase14hash",
            }
        ),
        encoding="utf-8",
    )
    portfolio_path = tmp_path / "current_positions.csv"
    portfolio_path.write_text(
        "account_id,week_end,symbol,quantity,market_price,market_value,weight,cash,source,paper_only,broker_submission_allowed\n"
        "acct,2026-04-24,QQQ,1.2,500.0,600.0,0.6,0.0,test,true,false\n"
        "acct,2026-04-24,BIL,4.0,100.0,400.0,0.4,0.0,test,true,false\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "phase15"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_phase15_sandbox.py",
            "--week-end",
            "2026-04-24",
            "--phase14-snapshot",
            str(phase14_path),
            "--portfolio-snapshot",
            str(portfolio_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=Path.cwd(),
    )

    summary = json.loads(
        (output_dir / "execution_sandbox_summary_latest.json").read_text(encoding="utf-8")
    )

    assert summary["signal_eligible"] is True
    assert summary["execution_allowed"] is True
