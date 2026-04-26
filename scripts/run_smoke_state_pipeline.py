from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qqq_cycle.core.calendar import load_fred_api_key
from qqq_cycle.core.covariance import RobustEWCov2D
from qqq_cycle.core.proto_online import initialize_prototypes_from_history, update_prototypes
from qqq_cycle.core.state_layer import compute_state_layer
from qqq_cycle.core.stress_layer import compute_stress_layer


def synthetic_inputs(n: int = 620) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2010-01-01", periods=n, freq="W-FRI")
    trend = np.linspace(0.0, 1.0, n)
    qqq = 100 * np.exp(np.cumsum(0.002 + 0.03 * rng.normal(size=n)))
    return pd.DataFrame(
        {
            "DFII10": 0.5 + 0.3 * np.sin(trend * 10) + 0.05 * rng.normal(size=n),
            "DGS2": 2.0 + 0.6 * np.sin(trend * 8) + 0.1 * rng.normal(size=n),
            "BAMLH0A0HYM2": 4.0 + 0.4 * np.cos(trend * 12) + 0.1 * rng.normal(size=n),
            "NFCI": -0.2 + 0.2 * np.sin(trend * 14) + 0.05 * rng.normal(size=n),
            "VIXCLS": 18.0 + 5.0 * np.sin(trend * 15) + rng.normal(size=n),
            "AI_GPR": np.maximum(0, 50 + 20 * np.sin(trend * 16) + 5 * rng.normal(size=n)),
            "USEPUINDXD": np.maximum(0, 90 + 30 * np.cos(trend * 11) + 8 * rng.normal(size=n)),
            "QQQ": qqq,
        },
        index=idx,
    )


def main() -> None:
    load_fred_api_key(ROOT / ".env")
    inputs = synthetic_inputs()
    state = compute_state_layer(inputs)
    theta = state[["H", "I"]].dropna()
    if len(theta) < 265:
        raise RuntimeError("synthetic data did not produce enough warm theta observations")

    cov = RobustEWCov2D()
    cov_state = cov.initialize_from_history(theta.iloc[:20].to_numpy())
    proto = initialize_prototypes_from_history(theta.iloc[:260].to_numpy())
    for t, (_, row) in enumerate(theta.iloc[:260].iterrows(), start=1):
        cov_state = cov.update(cov_state, row.to_numpy())

    for t, (_, row) in enumerate(theta.iloc[260:].head(20).iterrows(), start=261):
        prev_cov = cov_state.cov_reg.copy()
        prev_mean = cov_state.mean.copy()
        cov_state = cov.update(cov_state, row.to_numpy())
        proto = update_prototypes(
            proto,
            row.to_numpy(),
            cov_state.mean,
            prev_cov,
            cov_state.cov_reg,
            t,
        ).state
        del prev_mean

    stress = compute_stress_layer(state[["H", "I"]], state["E"])
    diagnostics = cov_state.last_diagnostics or {}
    required = [
        "maha",
        "huber_weight",
        "eigval_1",
        "eigval_2_raw",
        "eigval_2_reg",
        "condition_number_raw",
        "condition_number_reg",
        "eigval_2_was_floored",
    ]
    sample = {key: diagnostics.get(key) for key in required}
    print(
        json.dumps(
            {
                "rows": len(inputs),
                "theta_valid": len(theta),
                "stress_valid": int(stress.frame["s"].notna().sum()),
                "diagnostic_sample": sample,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
