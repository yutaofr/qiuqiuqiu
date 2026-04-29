from __future__ import annotations

import json
from pathlib import Path

import pytest

from qqq_cycle.portfolio.policy import load_portfolio_policy


POLICY_PATH = Path("configs/portfolio_policy_v1.yaml")


def _write_policy(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _base_policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def test_policy_loads() -> None:
    policy = load_portfolio_policy(POLICY_PATH)

    assert policy.paper_only is True
    assert policy.broker_submission_allowed is False
    assert policy.symbols == ("QQQ", "BIL")


def test_policy_rejects_non_paper_only(tmp_path: Path) -> None:
    payload = _base_policy()
    payload["paper_only"] = False

    with pytest.raises(ValueError):
        load_portfolio_policy(_write_policy(tmp_path, payload))


def test_policy_rejects_broker_submission(tmp_path: Path) -> None:
    payload = _base_policy()
    payload["broker_submission_allowed"] = True

    with pytest.raises(ValueError):
        load_portfolio_policy(_write_policy(tmp_path, payload))


def test_policy_requires_non_empty_universe(tmp_path: Path) -> None:
    payload = _base_policy()
    payload["universe"] = []

    with pytest.raises(ValueError):
        load_portfolio_policy(_write_policy(tmp_path, payload))


def test_policy_requires_weights_sum_to_one(tmp_path: Path) -> None:
    payload = _base_policy()
    payload["state_policy"]["strict_default"]["risk_mid"]["BIL"] = 0.3

    with pytest.raises(ValueError):
        load_portfolio_policy(_write_policy(tmp_path, payload))


def test_policy_rho_buckets_are_deterministic() -> None:
    policy = load_portfolio_policy(POLICY_PATH)

    assert policy.locate_rho_bucket(0.34) == "risk_low"
    assert policy.locate_rho_bucket(0.35) == "risk_mid"
    assert policy.locate_rho_bucket(0.65) == "risk_high"


def test_policy_rejects_bucket_gap(tmp_path: Path) -> None:
    payload = _base_policy()
    payload["rho_buckets"][1]["lower"] = 0.36

    with pytest.raises(ValueError):
        load_portfolio_policy(_write_policy(tmp_path, payload))


def test_policy_rejects_bucket_overlap(tmp_path: Path) -> None:
    payload = _base_policy()
    payload["rho_buckets"][1]["lower"] = 0.34

    with pytest.raises(ValueError):
        load_portfolio_policy(_write_policy(tmp_path, payload))
