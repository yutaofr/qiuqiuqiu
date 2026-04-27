"""Typed configuration loader for the QQQ cycle-state model.

The loader reads the checked-in v2.2 hyperparameter YAML as a startup-only
single source of truth. It intentionally supports only the small YAML subset
used by this repository: nested mappings, scalar numbers, and scalar lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DualMemoryConfig:
    """Dual-memory standardization windows in weekly units."""

    robust_window_weeks: int
    ew_half_life_weeks: int


@dataclass(frozen=True)
class CovarianceConfig:
    """Robust covariance recurrence parameters in weekly units."""

    half_life_weeks: int


@dataclass(frozen=True)
class DriftConfig:
    """Continuous drift blending thresholds in raw drift-probe units."""

    theta_lo: float
    theta_hi: float


@dataclass(frozen=True)
class MicroConfig:
    """Micro-layer IIR circuit breaker constants.

    Values are applied only to weekly h_t observations known at the decision
    timestamp; no future weekly or daily observations are read by this config.
    """

    iir_delta: float
    heal_threshold: float


@dataclass(frozen=True)
class RiskConfig:
    """Production rho_t constants and semantic state weights."""

    lambda_rho: float
    omega_state: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class ModelConfig:
    """Model v2.2 hyperparameters.

    Input: `model_v22.yaml` in this package, or an explicit path.
    Output: immutable typed configuration with dot access.
    Time/as-of semantics: configuration is static startup metadata and does
    not authorize using any market data beyond a decision timestamp.
    """

    warmup_weeks: int
    dual_memory: DualMemoryConfig
    covariance: CovarianceConfig
    drift: DriftConfig
    micro: MicroConfig
    risk: RiskConfig
    percentile_window_weeks: int
    noise_quantile: float


def _parse_scalar(raw: str) -> object:
    value = raw.strip()
    if value.startswith("[") and value.endswith("]"):
        items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
        return [_parse_scalar(item) for item in items]
    try:
        parsed = float(value)
    except ValueError:
        return value.strip("\"'")
    if parsed.is_integer() and "." not in value and "e" not in value.lower():
        return int(parsed)
    return parsed


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    parents: list[tuple[int, dict[str, Any]]] = [(-1, data)]
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"invalid indentation at {path}:{lineno}")
        if ":" not in raw_line:
            raise ValueError(f"invalid YAML line at {path}:{lineno}")
        key, raw_value = raw_line.strip().split(":", 1)
        while parents and indent <= parents[-1][0]:
            parents.pop()
        if not parents:
            raise ValueError(f"invalid parent indentation at {path}:{lineno}")
        parent = parents[-1][1]
        if raw_value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            parents.append((indent, child))
        else:
            parent[key] = _parse_scalar(raw_value)
    return data


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"model config missing mapping: {key}")
    return value


def load_config(path: str | Path | None = None) -> ModelConfig:
    """Load the immutable v2.2 model configuration.

    Args:
        path: Optional explicit YAML path. When omitted, reads the packaged
            `model_v22.yaml`.

    Returns:
        ModelConfig with dot-access dataclass fields.

    Raises:
        ValueError: If required keys are absent or malformed.
    """

    config_path = Path(path) if path is not None else Path(__file__).with_name("model_v22.yaml")
    raw = _parse_simple_yaml(config_path)
    dual_memory = _require_mapping(raw, "dual_memory")
    covariance = _require_mapping(raw, "covariance")
    drift = _require_mapping(raw, "drift")
    micro = _require_mapping(raw, "micro")
    risk = _require_mapping(raw, "risk")
    omega_state = tuple(float(value) for value in risk["omega_state"])
    if len(omega_state) != 5:
        raise ValueError("risk.omega_state must contain exactly five weights")
    return ModelConfig(
        warmup_weeks=int(raw["warmup_weeks"]),
        dual_memory=DualMemoryConfig(
            robust_window_weeks=int(dual_memory["robust_window_weeks"]),
            ew_half_life_weeks=int(dual_memory["ew_half_life_weeks"]),
        ),
        covariance=CovarianceConfig(half_life_weeks=int(covariance["half_life_weeks"])),
        drift=DriftConfig(
            theta_lo=float(drift["theta_lo"]),
            theta_hi=float(drift["theta_hi"]),
        ),
        micro=MicroConfig(
            iir_delta=float(micro["iir_delta"]),
            heal_threshold=float(micro["heal_threshold"]),
        ),
        risk=RiskConfig(
            lambda_rho=float(risk["lambda_rho"]),
            omega_state=omega_state,  # type: ignore[arg-type]
        ),
        percentile_window_weeks=int(raw["percentile_window_weeks"]),
        noise_quantile=float(raw["noise_quantile"]),
    )
