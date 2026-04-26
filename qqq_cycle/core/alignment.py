"""Point-in-time data alignment helpers."""

from __future__ import annotations

import pandas as pd

from qqq_cycle.core.calendar import validate_monotonic_index


def align_series_asof(
    series: pd.Series,
    decision_index: pd.DatetimeIndex,
    *,
    max_staleness: pd.Timedelta | None = None,
) -> pd.Series:
    """Align observations to decisions using only `observation_time <= decision`.

    Input:
        series: Timestamp-indexed observations.
        decision_index: Decision timestamps.
        max_staleness: Optional maximum age for the carried observation.
    Output:
        Series indexed by decision timestamps.
    As-of semantics:
        Uses pandas `merge_asof` backward direction only. Observations after a
        decision timestamp cannot influence that decision.
    """

    validate_monotonic_index(series.index, "series.index")
    validate_monotonic_index(decision_index, "decision_index")
    if len(decision_index) == 0:
        return pd.Series(dtype=float, index=decision_index, name=series.name)
    values = series.sort_index().rename("value").reset_index()
    values.columns = ["obs_time", "value"]
    decisions = pd.DataFrame({"decision_time": decision_index})
    aligned = pd.merge_asof(
        decisions,
        values,
        left_on="decision_time",
        right_on="obs_time",
        direction="backward",
        tolerance=max_staleness,
    )
    return pd.Series(
        aligned["value"].to_numpy(), index=decision_index, name=series.name
    )


def align_frame_asof(
    frame: pd.DataFrame,
    decision_index: pd.DatetimeIndex,
    *,
    max_staleness: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Align every column in a frame with strict backward as-of semantics."""

    return pd.DataFrame(
        {
            col: align_series_asof(frame[col], decision_index, max_staleness=max_staleness)
            for col in frame.columns
        },
        index=decision_index,
    )
