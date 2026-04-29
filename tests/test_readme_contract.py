from pathlib import Path


def test_readme_contains_current_status_and_safety_contract() -> None:
    readme = Path(__file__).resolve().parents[1].joinpath("README.md").read_text(
        encoding="utf-8"
    )

    required_snippets = [
        "This repository is not a live trading system.",
        "selected_scheme` | `degraded_backfill",
        "proof_strict_eligible` | `false",
        "Phase 15 is paper-only.",
        "broker_submission_allowed = false",
        "The system is not in `strict_recovery` for `2026-04-24`",
        "degraded_backfill` is the correct state when data validation passes but strict PIT proof fails.",
        "evaluate_publication_proof(...)",
    ]

    for snippet in required_snippets:
        assert snippet in readme
