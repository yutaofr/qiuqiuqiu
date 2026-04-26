import pandas as pd

from qqq_cycle.core.alignment import align_series_asof
from qqq_cycle.core.calendar import build_weekly_decision_index


def test_weekly_alignment_never_uses_future_observations() -> None:
    decisions = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-05 16:00"), pd.Timestamp("2024-01-12 16:00")]
    )
    observations = pd.Series(
        [10.0, 99.0, 12.0],
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-05 15:59"),
                pd.Timestamp("2024-01-05 16:01"),
                pd.Timestamp("2024-01-12 15:30"),
            ]
        ),
    )

    aligned = align_series_asof(observations, decisions)

    assert aligned.loc[decisions[0]] == 10.0
    assert aligned.loc[decisions[1]] == 12.0


def test_future_appends_do_not_change_past_aligned_values() -> None:
    decisions = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-05 16:00"), pd.Timestamp("2024-01-12 16:00")]
    )
    base = pd.Series(
        [10.0, 12.0],
        index=pd.DatetimeIndex(
            [pd.Timestamp("2024-01-05 15:59"), pd.Timestamp("2024-01-12 15:30")]
        ),
    )
    appended = pd.concat(
        [
            base,
            pd.Series(
                [1000.0],
                index=pd.DatetimeIndex([pd.Timestamp("2024-01-19 15:30")]),
            ),
        ]
    )

    aligned_base = align_series_asof(base, decisions)
    aligned_appended = align_series_asof(appended, decisions)

    pd.testing.assert_series_equal(aligned_base, aligned_appended)


def test_weekly_decision_index_uses_last_timestamp_per_week() -> None:
    dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-02 16:00"),
            pd.Timestamp("2024-01-05 16:00"),
            pd.Timestamp("2024-01-08 16:00"),
            pd.Timestamp("2024-01-12 16:00"),
        ]
    )

    decisions = build_weekly_decision_index(dates)

    assert list(decisions) == [
        pd.Timestamp("2024-01-05 16:00"),
        pd.Timestamp("2024-01-12 16:00"),
    ]
