"""Daily microstructure state primitives.

This module is intentionally split so early state handling can be tested
without price data. PIT adjusted-close lookups and rolling correlations are
added in later micro-layer functions; this file's core helpers only consume
daily constituent membership knowable at the daily as-of timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from qqq_cycle.data_contracts.pit_adjustment import DataNotAvailableError


class MicroLayerUnavailableError(RuntimeError):
    """Raised when mandatory point-in-time micro-layer data is unavailable."""


@dataclass(frozen=True)
class MicroScore:
    """Weekly micro fragility score before and after logistic mapping."""

    raw: float
    h_t: float


@dataclass(frozen=True)
class MicroIIRState:
    """Weekly micro IIR and breaker state.

    Inputs:
        h_t_lead_prev: Prior weekly lead micro fragility memory used by rho_t.
        heal_count: Consecutive strict low-micro weeks counted by the breaker.
        envelope_internal_state: Auditable copy of the current IIR envelope.
        breaker_internal_state: Human-readable breaker state.
        rho_update_state: Last real state source used for rho_t continuity.
        micro_state_frozen: Whether the current week intentionally froze state.

    Output/time semantics:
        State is updated only from current-week strict micro observations. A
        controlled degraded_backfill week returns the prior values unchanged.
    """

    h_t_lead_prev: float
    heal_count: int
    envelope_internal_state: float
    breaker_internal_state: str
    rho_update_state: str
    micro_state_frozen: bool = False

    @staticmethod
    def initial() -> "MicroIIRState":
        return MicroIIRState(
            h_t_lead_prev=0.0,
            heal_count=0,
            envelope_internal_state=0.0,
            breaker_internal_state="inactive",
            rho_update_state="initial",
            micro_state_frozen=False,
        )


@dataclass(frozen=True)
class MicroDailyState:
    """Point-in-time daily member state for micro-layer inputs.

    Inputs:
        member_ages: Trading-day membership ages observed through the current
            daily decision timestamp.
        present_members: Members present in the current PIT constituent
            snapshot.
        grace_missing_days: Missing members with consecutive missing trading
            day counts, capped by the configured grace period.
        grace_expiry: Last trade date through which each missing member remains
            in grace.
        smoothed_weights: Lagged smoothed weights from prior known holdings.
        data_contaminated: Whether the current week must carry c_t^data = 1.

    Output/time semantics:
        State is a pure as-of snapshot. Missing members in grace keep their age
        counter frozen and are excluded from V^20/V^60 until they reappear.
    """

    member_ages: Mapping[str, int]
    present_members: frozenset[str]
    grace_missing_days: Mapping[str, int]
    grace_expiry: Mapping[str, pd.Timestamp]
    smoothed_weights: Mapping[str, float]
    data_contaminated: bool = False

    @staticmethod
    def empty() -> "MicroDailyState":
        """Return an empty PIT daily micro-layer state."""

        return MicroDailyState(
            member_ages=MappingProxyType({}),
            present_members=frozenset(),
            grace_missing_days=MappingProxyType({}),
            grace_expiry=MappingProxyType({}),
            smoothed_weights=MappingProxyType({}),
            data_contaminated=False,
        )

    @property
    def grace_members(self) -> frozenset[str]:
        """Members currently in the <=3 trading-day grace period."""

        return frozenset(self.grace_missing_days)

    def with_smoothed_weights(self, weights: Mapping[str, float]) -> "MicroDailyState":
        """Return a copy with lagged smoothed weights attached.

        Inputs are prior-day weights, not current-day prices; callers must only
        pass values knowable by the current daily as-of timestamp.
        """

        return MicroDailyState(
            member_ages=self.member_ages,
            present_members=self.present_members,
            grace_missing_days=self.grace_missing_days,
            grace_expiry=self.grace_expiry,
            smoothed_weights=MappingProxyType({k: float(v) for k, v in weights.items()}),
            data_contaminated=self.data_contaminated,
        )


@dataclass(frozen=True)
class MissingWeightDecision:
    """Giant missing-weight guard decision for a daily micro recompute."""

    hold_micro_recompute: bool
    data_contaminated: bool
    missing_weight: float
    threshold: float


def update_micro_daily_state(
    state: MicroDailyState,
    current_members: set[str] | frozenset[str],
    trade_date: pd.Timestamp,
    *,
    grace_period_days: int = 3,
) -> MicroDailyState:
    """Advance daily member ages and grace-period status.

    Args:
        state: Previous daily state known before `trade_date`.
        current_members: PIT constituent snapshot members for `trade_date`.
        trade_date: Current trade date whose membership is knowable as of the
            daily decision timestamp.
        grace_period_days: Maximum consecutive missing trading days for which
            an existing member's counter is frozen rather than reset.

    Returns:
        New immutable state. Members in grace are excluded from mature sets and
        set `data_contaminated=True` for the current weekly aggregation.
    """

    if grace_period_days < 0:
        raise ValueError("grace_period_days must be non-negative")
    current = frozenset(current_members)
    previous_tracked = set(state.member_ages) | set(state.present_members) | set(state.grace_missing_days)
    ages: dict[str, int] = {}
    grace_missing_days: dict[str, int] = {}
    grace_expiry: dict[str, pd.Timestamp] = {}
    trade_ts = pd.Timestamp(trade_date)

    for ticker in sorted(previous_tracked | set(current)):
        previous_age = int(state.member_ages.get(ticker, 0))
        if ticker in current:
            ages[ticker] = previous_age + 1
            continue

        if previous_age <= 0:
            continue
        missing_days = int(state.grace_missing_days.get(ticker, 0)) + 1
        if missing_days <= grace_period_days:
            ages[ticker] = previous_age
            grace_missing_days[ticker] = missing_days
            grace_expiry[ticker] = trade_ts + pd.tseries.offsets.BDay(
                grace_period_days - missing_days
            )

    contaminated = bool(grace_missing_days)
    return MicroDailyState(
        member_ages=MappingProxyType(ages),
        present_members=current,
        grace_missing_days=MappingProxyType(grace_missing_days),
        grace_expiry=MappingProxyType(grace_expiry),
        smoothed_weights=state.smoothed_weights,
        data_contaminated=contaminated,
    )


def matured_member_sets(state: MicroDailyState) -> tuple[frozenset[str], frozenset[str]]:
    """Return V_tau^20 and V_tau^60 mature member sets.

    Grace-period members are excluded even when their frozen age meets maturity.
    Only members present in the current PIT snapshot can enter either set.
    """

    eligible = state.present_members - state.grace_members
    v20 = frozenset(
        ticker for ticker in eligible if int(state.member_ages.get(ticker, 0)) >= 20
    )
    v60 = frozenset(
        ticker for ticker in eligible if int(state.member_ages.get(ticker, 0)) >= 60
    )
    return v20, v60


def should_hold_for_giant_missing_weight(state: MicroDailyState) -> MissingWeightDecision:
    """Evaluate the §9.5 giant missing-weight threshold.

    The missing weight is the prior-day smoothed lagged weight sum for grace
    members. The threshold is 0.5 times the fifth-largest smoothed weight known
    before the current day. If fewer than five weights are available, the guard
    cannot be evaluated and does not hold the recompute.
    """

    weights = {ticker: float(weight) for ticker, weight in state.smoothed_weights.items()}
    missing_weight = float(sum(weights.get(ticker, 0.0) for ticker in state.grace_members))
    ranked = sorted(weights.values(), reverse=True)
    threshold = float("inf") if len(ranked) < 5 else 0.5 * float(ranked[4])
    hold = missing_weight > threshold
    return MissingWeightDecision(
        hold_micro_recompute=hold,
        data_contaminated=bool(state.data_contaminated or hold),
        missing_weight=missing_weight,
        threshold=threshold,
    )


def compute_smoothed_weights(
    previous_smoothed: Mapping[str, float],
    lagged_weights: Mapping[str, float],
    *,
    is_rule_window: bool,
    rho_w: float = 2 ** (-1 / 5),
) -> dict[str, float]:
    """Compute §9.6 smoothed lagged weights.

    Args:
        previous_smoothed: Prior smoothed lagged weights known before today.
        lagged_weights: Previous trading day's PIT holdings weights.
        is_rule_window: True for rule/rebalance windows where updates freeze.
        rho_w: Five-trading-day half-life coefficient.

    Returns:
        Ticker-to-weight mapping. Rule windows return the prior smoothed weights
        unchanged; otherwise the union of previous and lagged tickers is updated.
    """

    if not 0.0 < rho_w < 1.0:
        raise ValueError("rho_w must be in (0, 1)")
    if is_rule_window:
        return {ticker: float(weight) for ticker, weight in previous_smoothed.items()}
    tickers = sorted(set(previous_smoothed) | set(lagged_weights))
    return {
        ticker: rho_w * float(previous_smoothed.get(ticker, 0.0))
        + (1.0 - rho_w) * float(lagged_weights.get(ticker, 0.0))
        for ticker in tickers
    }


def compute_breadth(
    *,
    members: frozenset[str],
    smoothed_weights: Mapping[str, float],
    trade_date: pd.Timestamp,
    pit_engine: object | None,
    eps: float = 1e-12,
) -> float:
    """Compute §9.7 weighted breadth collapse b_tau.

    The function obtains each MA20 window through
    `pit_engine.get_adjusted_window(ticker, trade_date, 20, asof=trade_date)`.
    If a PIT adjusted window cannot be provided, the micro layer fails closed
    with `MicroLayerUnavailableError`.
    """

    if pit_engine is None or not hasattr(pit_engine, "get_adjusted_window"):
        raise MicroLayerUnavailableError("PITAdjustmentEngine with get_adjusted_window is required")
    weighted_above_ma = 0.0
    total_weight = 0.0
    date = pd.Timestamp(trade_date)
    for ticker in sorted(members):
        weight = float(smoothed_weights.get(ticker, 0.0))
        if weight <= 0.0:
            continue
        try:
            window = pit_engine.get_adjusted_window(ticker, date, 20, asof=date)
        except DataNotAvailableError as exc:
            raise MicroLayerUnavailableError(str(exc)) from exc
        series = pd.Series(window, dtype=float).dropna()
        if len(series) < 20:
            raise MicroLayerUnavailableError(
                f"need 20 PIT adjusted closes for {ticker}; got {len(series)}"
            )
        total_weight += weight
        latest = float(series.iloc[-1])
        ma20 = float(series.tail(20).mean())
        weighted_above_ma += weight * float(latest > ma20)
    if total_weight <= eps:
        return float("nan")
    breadth = float(1.0 - weighted_above_ma / (total_weight + eps))
    if abs(breadth) <= 1e-10:
        return 0.0
    return float(np.clip(breadth, 0.0, 1.0))


def compute_correlation_concentration(
    *,
    members: frozenset[str],
    smoothed_weights: Mapping[str, float],
    price_windows: Mapping[str, pd.Series],
    eps: float = 1e-12,
) -> float:
    """Compute §9.8 weighted 60-day correlation concentration c_tau.

    Inputs are already point-in-time adjusted price windows ending at the daily
    as-of timestamp. Returns the normalized weighted off-diagonal correlation.
    """

    tickers = [
        ticker
        for ticker in sorted(members)
        if ticker in price_windows and float(smoothed_weights.get(ticker, 0.0)) > 0.0
    ]
    if len(tickers) < 2:
        return float("nan")
    prices = pd.DataFrame({ticker: pd.Series(price_windows[ticker], dtype=float) for ticker in tickers})
    returns = prices.pct_change(fill_method=None).dropna(how="any")
    if len(returns) < 2:
        return float("nan")
    corr = returns.corr().to_numpy(dtype=float)
    if not np.isfinite(corr).all():
        return float("nan")
    raw_weights = np.array([float(smoothed_weights[ticker]) for ticker in tickers], dtype=float)
    total = float(raw_weights.sum())
    if total <= eps:
        return float("nan")
    weights = raw_weights / (total + eps)
    numerator = float(weights.T @ (corr - np.eye(len(tickers))) @ weights)
    denominator = float(1.0 - np.dot(weights, weights) + eps)
    return numerator / denominator


def weekly_median_micro(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily b_tau/c_tau to Friday-labeled weekly medians.

    Only rows whose daily timestamps fall inside each weekly period contribute
    to that week's median; no future-week values are read.
    """

    required = {"b_tau", "c_tau"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily frame missing columns: {sorted(missing)}")
    frame = daily.sort_index()[["b_tau", "c_tau"]].copy()
    weekly = frame.resample("W-FRI", label="right", closed="right").median()
    return weekly.rename(columns={"b_tau": "b_wk", "c_tau": "c_wk"})


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    vals = values[finite]
    wts = weights[finite]
    if vals.size == 0:
        return float("nan")
    order = np.argsort(vals)
    vals = vals[order]
    wts = wts[order]
    cutoff = 0.5 * float(wts.sum())
    return float(vals[np.searchsorted(np.cumsum(wts), cutoff, side="left")])


def z_wrob_156(
    series: pd.Series,
    *,
    weights: pd.Series,
    window: int = 156,
    eps: float = 1e-12,
) -> pd.Series:
    """Return §9.10 weighted robust z-score with prior-window baselines.

    Each output at week t uses only `series[t-window:t-1]` and matching rule
    weights from that same historical window. The current observation never
    enters its own median/MAD baseline.
    """

    if window < 1:
        raise ValueError("window must be >= 1")
    values = pd.Series(series, dtype=float)
    w = pd.Series(weights, dtype=float).reindex(values.index)
    out = pd.Series(np.nan, index=values.index, dtype=float)
    arr = values.to_numpy(dtype=float)
    warr = w.to_numpy(dtype=float)
    for i, current in enumerate(arr):
        if not np.isfinite(current):
            continue
        start = max(0, i - window)
        hist = arr[start:i]
        hist_w = warr[start:i]
        finite = np.isfinite(hist) & np.isfinite(hist_w) & (hist_w > 0.0)
        if not finite.any():
            continue
        median = _weighted_median(hist, hist_w)
        mad = _weighted_median(np.abs(hist - median), hist_w)
        out.iloc[i] = (current - median) / (1.4826 * mad + eps)
    return out


def _logistic(x: float) -> float:
    if x >= 0.0:
        return float(1.0 / (1.0 + np.exp(-x)))
    exp_x = float(np.exp(x))
    return float(exp_x / (1.0 + exp_x))


def compute_micro_score(b_tilde: float, c_tilde: float) -> MicroScore:
    """Compute §9.11 raw micro score and bounded h_t.

    Inputs are weekly z_wrob outputs for breadth and correlation concentration
    known at the weekly decision timestamp.
    """

    raw = 0.5 * float(b_tilde) + 0.5 * float(c_tilde)
    return MicroScore(raw=raw, h_t=_logistic(raw))


def iir_envelope_with_breaker(
    h_t: float,
    h_t_minus1: float,
    h_t_minus2: float,
    x_lead_prev: float,
    delta: float = 0.9,
    theta_heal: float = 0.25,
) -> float:
    """Return x_t_lead per architecture spec §6.3 / model spec §9.11.

    NOTE: This is a standalone pure function that pre-shifts h_t by -0.5
    (output x_lead ∈ [0, 0.5]).  It is NOT the IIR implementation used in the
    production pipeline.  pipeline.py runs the IIR on raw h_t ∈ [0, 1] inline
    in the weekly loop (Option B), consistent with compute_risk_score which
    applies the shift internally via n_t = 2*max(h_t_lead - 0.5, 0).  Composing
    this function with compute_risk_score would double-shift and produce n_t ≈ 0.
    """

    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be in [0, 1]")
    heal = h_t < theta_heal and h_t_minus1 < theta_heal and h_t_minus2 < theta_heal
    if heal:
        return 0.0
    x_t = max(float(h_t) - 0.5, 0.0)
    return float(max(x_t, delta * float(x_lead_prev)))


def update_weekly_micro_iir_state(
    prior: MicroIIRState,
    *,
    h_t_raw: float | None,
    backfill_mode: str | None = None,
    delta: float = 0.9,
    theta_heal: float = 0.25,
    heal_weeks: int = 3,
) -> MicroIIRState:
    """Advance or freeze weekly micro IIR state.

    A `degraded_backfill` week freezes h_t_lead_prev, heal_count, envelope
    internals, breaker internals, and the rho update state. No passive decay,
    empty-observation decay, or breaker transition is applied.
    """

    if backfill_mode == "degraded_backfill":
        return MicroIIRState(
            h_t_lead_prev=prior.h_t_lead_prev,
            heal_count=prior.heal_count,
            envelope_internal_state=prior.envelope_internal_state,
            breaker_internal_state=prior.breaker_internal_state,
            rho_update_state=prior.rho_update_state,
            micro_state_frozen=True,
        )
    if h_t_raw is None or not np.isfinite(float(h_t_raw)):
        return MicroIIRState(
            h_t_lead_prev=prior.h_t_lead_prev,
            heal_count=prior.heal_count,
            envelope_internal_state=prior.envelope_internal_state,
            breaker_internal_state=prior.breaker_internal_state,
            rho_update_state=prior.rho_update_state,
            micro_state_frozen=False,
        )
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be in [0, 1]")
    if heal_weeks < 1:
        raise ValueError("heal_weeks must be >= 1")

    h = float(h_t_raw)
    h_t_lead = max(h, float(delta) * float(prior.h_t_lead_prev))
    heal_count = int(prior.heal_count)
    breaker_state = "inactive"
    if h < theta_heal:
        heal_count += 1
        breaker_state = "healing"
        if heal_count >= heal_weeks:
            h_t_lead = h
            heal_count = 0
            breaker_state = "reset"
    else:
        heal_count = 0

    return MicroIIRState(
        h_t_lead_prev=h_t_lead,
        heal_count=heal_count,
        envelope_internal_state=h_t_lead,
        breaker_internal_state=breaker_state,
        rho_update_state="strict_observation",
        micro_state_frozen=False,
    )
