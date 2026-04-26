"""Calendar and environment helpers with point-in-time week semantics."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


def load_fred_api_key(env_path: str | Path = ".env") -> str:
    """Return FRED API key from `.env` without hardcoding or printing it.

    Input:
        env_path: Path to an environment file.
    Output:
        Non-empty FRED API key string.
    Time semantics:
        No market data is loaded; this only reads process configuration.
    """

    load_dotenv(env_path)
    key = os.getenv("FRED_API_KEY", "")
    if not key:
        raise RuntimeError("FRED_API_KEY is missing from .env")
    return key


def validate_monotonic_index(index: pd.DatetimeIndex, name: str = "index") -> None:
    """Validate a timestamp index before PIT alignment.

    Raises:
        ValueError if the index is not a DatetimeIndex, has nulls, duplicates,
        or is not monotonic increasing.
    """

    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{name} must be a pandas DatetimeIndex")
    if index.hasnans:
        raise ValueError(f"{name} contains NaT values")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} must be monotonic increasing")
    if not index.is_unique:
        raise ValueError(f"{name} must be unique")


def build_weekly_decision_index(timestamps: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the last available timestamp in each Friday-ending week.

    Input:
        timestamps: Observation timestamps already constrained to knowable data.
    Output:
        WeekIndex where each item is that week's decision timestamp.
    As-of semantics:
        The decision timestamp is selected from existing observations only; no
        synthetic future week-end is introduced.
    """

    validate_monotonic_index(timestamps, "timestamps")
    if len(timestamps) == 0:
        return pd.DatetimeIndex([])
    series = pd.Series(timestamps, index=timestamps)
    weekly = series.groupby(timestamps.to_period("W-FRI")).max()
    return pd.DatetimeIndex(weekly.to_list())
