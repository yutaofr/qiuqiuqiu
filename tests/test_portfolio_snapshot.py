from __future__ import annotations

from pathlib import Path

import pytest

from qqq_cycle.portfolio.policy import load_portfolio_policy
from qqq_cycle.portfolio.portfolio_snapshot import load_portfolio_snapshot


POLICY_PATH = Path("configs/portfolio_policy_v1.yaml")


def _write_snapshot(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _base_csv() -> str:
    return (
        "account_id,week_end,symbol,quantity,market_price,market_value,weight,cash,source,paper_only,broker_submission_allowed\n"
        "acct,2026-04-24,QQQ,1.2,500.0,600.0,0.6,0.0,test,true,false\n"
        "acct,2026-04-24,BIL,4.0,100.0,400.0,0.4,0.0,test,true,false\n"
    )


def test_load_valid_snapshot(tmp_path: Path) -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    snapshot = load_portfolio_snapshot(_write_snapshot(tmp_path / "snapshot.csv", _base_csv()), policy)

    assert snapshot.paper_only is True
    assert snapshot.broker_submission_allowed is False
    assert snapshot.nav == pytest.approx(1000.0)


def test_snapshot_requires_paper_only(tmp_path: Path) -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    body = _base_csv().replace("true,false", "false,false", 1)
    with pytest.raises(ValueError):
        load_portfolio_snapshot(_write_snapshot(tmp_path / "snapshot.csv", body), policy)


def test_snapshot_requires_broker_submission_false(tmp_path: Path) -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    body = _base_csv().replace("true,false", "true,true", 1)
    with pytest.raises(ValueError):
        load_portfolio_snapshot(_write_snapshot(tmp_path / "snapshot.csv", body), policy)


def test_snapshot_requires_weights_sum_near_one(tmp_path: Path) -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    body = _base_csv().replace(",0.4,0.0,", ",0.2,0.0,")
    with pytest.raises(ValueError):
        load_portfolio_snapshot(_write_snapshot(tmp_path / "snapshot.csv", body), policy)


def test_snapshot_requires_non_negative_cash(tmp_path: Path) -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    body = _base_csv().replace(",0.0,test", ",-1.0,test")
    with pytest.raises(ValueError):
        load_portfolio_snapshot(_write_snapshot(tmp_path / "snapshot.csv", body), policy)


def test_snapshot_rejects_unknown_symbol(tmp_path: Path) -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    body = _base_csv().replace("BIL", "TLT")
    with pytest.raises(ValueError):
        load_portfolio_snapshot(_write_snapshot(tmp_path / "snapshot.csv", body), policy)


def test_snapshot_rejects_negative_quantity_when_shorting_disabled(tmp_path: Path) -> None:
    policy = load_portfolio_policy(POLICY_PATH)
    body = _base_csv().replace("1.2", "-1.2", 1)
    with pytest.raises(ValueError):
        load_portfolio_snapshot(_write_snapshot(tmp_path / "snapshot.csv", body), policy)
