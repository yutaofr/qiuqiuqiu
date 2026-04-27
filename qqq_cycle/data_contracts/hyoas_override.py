"""Rigid CSV contract for HYOAS override (BAMLH0A0HYM2)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 265 theta minimum + 260 dual-memory warmup buffer
MIN_WARMUP_ROWS = 525

@dataclass(frozen=True)
class HyoasOverrideContract:
    """Individual record for HYOAS override."""
    trade_date: pd.Timestamp
    value: float
    source_name: str
    source_timestamp: pd.Timestamp
    license_tag: str
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.source_name:
            raise ValueError("source_name is required")
        if not self.license_tag:
            raise ValueError("license_tag is required")
        if not np.isfinite(self.value):
            raise ValueError("value must be finite")

class HyoasOverrideStore:
    """Validator and loader for licensed HYOAS CSV overrides."""

    REQUIRED_COLUMNS = {
        "trade_date",
        "value",
        "source_name",
        "source_timestamp",
        "license_tag",
    }

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"HYOAS override file not found: {self.path}")
        
        raw = pd.read_csv(self.path)
        normalized = {str(col).strip().lower(): col for col in raw.columns}
        missing = self.REQUIRED_COLUMNS.difference(normalized)
        if missing:
            raise ValueError(f"HYOAS override CSV missing required columns: {sorted(missing)}")
        
        # Standardize columns
        frame = raw.rename(columns={original: key for key, original in normalized.items()})
        
        self._frame = pd.DataFrame({
            "trade_date": pd.to_datetime(frame["trade_date"]),
            "value": pd.to_numeric(frame["value"], errors="coerce"),
            "source_name": frame["source_name"].astype(str),
            "source_timestamp": pd.to_datetime(frame["source_timestamp"]),
            "license_tag": frame["license_tag"].astype(str),
        })
        if "notes" in normalized:
            self._frame["notes"] = frame["notes"].astype(str)
        else:
            self._frame["notes"] = None

        # Strict validation
        if self._frame["trade_date"].isnull().any():
            raise ValueError("trade_date contains nulls or unparseable values")

        if not self._frame["trade_date"].is_monotonic_increasing:
             raise ValueError("trade_date must be strictly increasing")

        if self._frame["trade_date"].duplicated().any():
            raise ValueError("HYOAS override CSV contains duplicate trade_date")

        # Check for strict increase (no same dates allowed) - handled by monotonic + duplicated


        if self._frame["value"].isnull().any() or not np.isfinite(self._frame["value"]).all():
            raise ValueError("HYOAS override CSV contains non-numeric or non-finite value")

    def to_series(self) -> pd.Series:
        """Return BAMLH0A0HYM2 series for replay."""
        return pd.Series(
            self._frame["value"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(self._frame["trade_date"]),
            name="BAMLH0A0HYM2"
        )

    def generate_manifest(self) -> dict[str, Any]:
        """Generate metadata for auditing the override."""
        row_count = len(self._frame)
        coverage_ok = row_count >= MIN_WARMUP_ROWS
        
        return {
            "source_name": self._frame["source_name"].iloc[0] if row_count > 0 else "unknown",
            "source_timestamp": self._frame["source_timestamp"].iloc[0].isoformat() if row_count > 0 else None,
            "min_date": self._frame["trade_date"].min().strftime("%Y-%m-%d") if row_count > 0 else None,
            "max_date": self._frame["trade_date"].max().strftime("%Y-%m-%d") if row_count > 0 else None,
            "row_count": int(row_count),
            "coverage_ok": bool(coverage_ok),
            "license_tag": self._frame["license_tag"].iloc[0] if row_count > 0 else "unknown",
            "reason_if_rejected": None if coverage_ok else f"Insufficient coverage: {row_count} < {MIN_WARMUP_ROWS} rows"
        }
