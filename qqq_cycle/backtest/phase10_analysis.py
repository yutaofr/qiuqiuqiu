"""Phase 10: Regime separation and signal predictive power diagnostics.

Two sequential layers:
  B — Confirm k_hat_t labels correspond to distinct realized market environments.
  A — Confirm h_t / rho_t / s_t / I_t carry forward-looking information.

No position sizing, no P&L, no strategy logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGNALS = ["h_t", "rho_t", "s_t", "I_t"]
HORIZONS = [1, 4, 8]
REALIZED_METRICS = ["r1w", "R4w", "sigma4w", "mdd8w"]
FORWARD_TARGETS = ["fwd_ret", "fwd_vol", "fwd_mdd"]

HIGH_CONF_THRESHOLD = 0.60  # max(p_t) >= this for high-confidence subsample


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_strict_rows(pipeline_csv: str | Path) -> pd.DataFrame:
    """Load and filter strict rows from pipeline output CSV.

    Returns DataFrame with parsed p_t (as list) and numeric signals.
    """
    df = pd.read_csv(pipeline_csv, parse_dates=["week_end"])
    df = df[df["mode"] == "strict"].copy()
    df = df[df["strict_contracts_satisfied"] == True].copy()  # noqa: E712

    df["p_t_list"] = df["p_t"].apply(
        lambda s: json.loads(s) if isinstance(s, str) else []
    )
    df["max_p_t"] = df["p_t_list"].apply(lambda lst: max(lst) if lst else np.nan)
    df["k_hat_t"] = df["k_hat_t"].astype(int)

    for col in SIGNALS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("week_end").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# QQQ price helpers
# ---------------------------------------------------------------------------


def fetch_qqq_weekly(
    start: str,
    end: str,
    cache_path: str | Path,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return QQQ weekly adjusted-close aligned to Friday (or last trading day).

    Downloads from yfinance and caches to cache_path.
    Columns: week_end (date), qqq_adj_close (float).
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path, parse_dates=["week_end"])
        return df

    import yfinance as yf

    # Extend start back slightly to cover 8-week trailing realized metrics
    fetch_start = pd.Timestamp(start) - pd.DateOffset(weeks=10)
    raw = yf.download(
        "QQQ",
        start=fetch_start.strftime("%Y-%m-%d"),
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned empty data for QQQ")

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    daily = raw[["Close"]].rename(columns={"Close": "qqq_adj_close"})
    daily.index.name = "date"
    daily.index = pd.to_datetime(daily.index)

    # Resample to weekly: last trading day of each ISO week
    weekly = daily.resample("W-FRI").last().dropna()
    weekly.index.name = "week_end"
    weekly = weekly.reset_index()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(cache_path, index=False)
    return weekly


def align_qqq_to_pipeline(
    pipeline_df: pd.DataFrame,
    qqq_weekly: pd.DataFrame,
) -> pd.DataFrame:
    """Merge QQQ weekly closes onto pipeline week_end dates (nearest prior week).

    Returns pipeline_df with qqq_adj_close column added.
    """
    qqq = qqq_weekly.set_index("week_end").sort_index()
    result = pipeline_df.copy()
    result["qqq_adj_close"] = result["week_end"].apply(
        lambda d: _lookup_nearest_prior(qqq["qqq_adj_close"], d)
    )
    return result


def _lookup_nearest_prior(series: pd.Series, date: pd.Timestamp) -> float:
    idx = series.index[series.index <= date]
    if idx.empty:
        return np.nan
    return series[idx[-1]]


# ---------------------------------------------------------------------------
# Realized metric computation (backward-looking, no lookahead)
# ---------------------------------------------------------------------------


def compute_realized_metrics(
    qqq_daily: pd.DataFrame,
    week_ends: pd.Series,
) -> pd.DataFrame:
    """Compute backward-looking realized metrics at each week_end date.

    Args:
        qqq_daily: DataFrame with DatetimeIndex and column 'qqq_adj_close'.
        week_ends: Series of Timestamps (pipeline week_end dates).

    Returns DataFrame indexed by week_end with columns:
        r1w, R4w, sigma4w, mdd8w
    """
    price = qqq_daily["qqq_adj_close"].sort_index()
    records = []
    for wk in week_ends:
        records.append(_realized_at(price, wk))
    df = pd.DataFrame(records, index=week_ends)
    df.index.name = "week_end"
    return df


def _realized_at(price: pd.Series, date: pd.Timestamp) -> dict:
    # Prices up to and including date
    hist = price[price.index <= date]
    if len(hist) < 2:
        return {m: np.nan for m in REALIZED_METRICS}

    p_t = hist.iloc[-1]

    # 1-week return
    prev_1w = hist.index[hist.index <= date - pd.Timedelta(days=5)]
    r1w = np.log(p_t / hist[prev_1w[-1]]) if len(prev_1w) else np.nan

    # 4-week trailing return
    prev_4w = hist.index[hist.index <= date - pd.Timedelta(days=25)]
    R4w = np.log(p_t / hist[prev_4w[-1]]) if len(prev_4w) else np.nan

    # 4-week realized volatility (annualise: *sqrt(52))
    window_4w = hist[hist.index > date - pd.Timedelta(days=28)]
    if len(window_4w) >= 2:
        log_rets = np.log(window_4w / window_4w.shift(1)).dropna()
        sigma4w = float(log_rets.std() * np.sqrt(52))
    else:
        sigma4w = np.nan

    # 8-week max drawdown
    window_8w = hist[hist.index > date - pd.Timedelta(days=56)]
    mdd8w = _max_drawdown(window_8w) if len(window_8w) >= 2 else np.nan

    return {"r1w": r1w, "R4w": R4w, "sigma4w": sigma4w, "mdd8w": mdd8w}


def _max_drawdown(price_series: pd.Series) -> float:
    """Max drawdown (positive number, e.g. 0.15 = 15% drawdown)."""
    rolling_peak = price_series.cummax()
    drawdowns = (price_series - rolling_peak) / rolling_peak
    return float(-drawdowns.min())


# ---------------------------------------------------------------------------
# Forward metric computation
# ---------------------------------------------------------------------------


def compute_forward_metrics(
    qqq_daily: pd.DataFrame,
    week_ends: pd.Series,
    horizons: list[int] = HORIZONS,
) -> pd.DataFrame:
    """Compute forward-looking metrics at each week_end date.

    For horizon h (in weeks), computes metrics over the window (week_end, week_end + h weeks].

    Returns DataFrame with columns: fwd_ret_{h}w, fwd_vol_{h}w, fwd_mdd_{h}w
    """
    price = qqq_daily["qqq_adj_close"].sort_index()
    all_records = []
    for wk in week_ends:
        rec = {"week_end": wk}
        for h in horizons:
            fwd = _forward_window(price, wk, h)
            rec[f"fwd_ret_{h}w"] = fwd["ret"]
            rec[f"fwd_vol_{h}w"] = fwd["vol"]
            rec[f"fwd_mdd_{h}w"] = fwd["mdd"]
        all_records.append(rec)
    return pd.DataFrame(all_records).set_index("week_end")


def _forward_window(price: pd.Series, date: pd.Timestamp, h_weeks: int) -> dict:
    """Metrics over (date, date + h_weeks*7 days]."""
    p0 = price[price.index <= date]
    if p0.empty:
        return {"ret": np.nan, "vol": np.nan, "mdd": np.nan}
    p_start = p0.iloc[-1]

    end_date = date + pd.Timedelta(days=h_weeks * 7)
    fwd = price[(price.index > date) & (price.index <= end_date)]
    if fwd.empty:
        return {"ret": np.nan, "vol": np.nan, "mdd": np.nan}

    ret = float(np.log(fwd.iloc[-1] / p_start))
    log_rets = np.log(fwd / fwd.shift(1)).dropna()
    vol = float(log_rets.std() * np.sqrt(52)) if len(log_rets) >= 2 else np.nan
    mdd = _max_drawdown(pd.concat([pd.Series([p_start]), fwd]))

    return {"ret": ret, "vol": vol, "mdd": mdd}


# ---------------------------------------------------------------------------
# Part B — Regime Separation
# ---------------------------------------------------------------------------


def run_regime_separation(df: pd.DataFrame) -> dict:
    """Run full Part B regime separation analysis.

    Args:
        df: merged DataFrame with k_hat_t and realized metric columns.

    Returns dict with keys: summary_records, tests, passed.
    """
    summary_records = []
    tests = {}

    for metric in REALIZED_METRICS:
        if metric not in df.columns:
            continue
        col = df[metric].dropna()
        groups = [df.loc[col.index][metric][df.loc[col.index]["k_hat_t"] == s].values
                  for s in range(5)]
        groups = [g for g in groups if len(g) >= 3]

        # Descriptive stats per state
        for s in range(5):
            sub = df[df["k_hat_t"] == s][metric].dropna()
            summary_records.append({
                "metric": metric,
                "state": s,
                "n_obs": len(sub),
                "median": float(sub.median()) if len(sub) else np.nan,
                "q25": float(sub.quantile(0.25)) if len(sub) else np.nan,
                "q75": float(sub.quantile(0.75)) if len(sub) else np.nan,
            })

        if len(groups) < 2:
            tests[metric] = {"kw_stat": np.nan, "kw_pvalue": np.nan, "pairwise": []}
            continue

        kw_stat, kw_pvalue = stats.kruskal(*groups)
        pairwise = _dunn_pairwise_holm(df, metric)
        tests[metric] = {
            "kw_stat": float(kw_stat),
            "kw_pvalue": float(kw_pvalue),
            "pairwise": pairwise,
        }

    passed = _check_regime_separation_pass(tests)
    return {"summary_records": summary_records, "tests": tests, "passed": passed}


def _dunn_pairwise_holm(df: pd.DataFrame, metric: str) -> list[dict]:
    """Dunn pairwise test with Holm-Bonferroni correction.

    Returns list of dicts with keys: state_i, state_j, z_stat, pvalue_raw,
    pvalue_holm, significant, cliffs_delta.
    """
    all_values = df[metric].dropna().values
    all_ranks = stats.rankdata(all_values)
    idx = df[metric].dropna().index

    # Map original index positions to ranks
    rank_series = pd.Series(all_ranks, index=idx)

    n_total = len(all_values)

    # Tie correction
    _, counts = np.unique(all_values, return_counts=True)
    tie_correction = np.sum(counts**3 - counts) / (12.0 * n_total * (n_total - 1))

    state_ranks: dict[int, np.ndarray] = {}
    state_vals: dict[int, np.ndarray] = {}
    for s in range(5):
        mask = df.loc[idx, "k_hat_t"] == s
        state_ranks[s] = rank_series[mask[mask].index].values
        state_vals[s] = df.loc[idx][metric][mask].values

    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    raw_results = []
    for si, sj in pairs:
        ri, rj = state_ranks[si], state_ranks[sj]
        if len(ri) < 2 or len(rj) < 2:
            raw_results.append((si, sj, np.nan, 1.0, cliffs_delta(state_vals[si], state_vals[sj])))
            continue
        ni, nj = len(ri), len(rj)
        mean_ri, mean_rj = ri.mean(), rj.mean()
        se = np.sqrt(
            (n_total * (n_total + 1) / 12.0 - tie_correction) * (1.0 / ni + 1.0 / nj)
        )
        z = (mean_ri - mean_rj) / se if se > 0 else 0.0
        p_raw = 2.0 * stats.norm.sf(abs(z))
        cd = cliffs_delta(state_vals[si], state_vals[sj])
        raw_results.append((si, sj, float(z), float(p_raw), cd))

    # Holm-Bonferroni correction
    m = len(raw_results)
    order = np.argsort([r[3] for r in raw_results])
    holm_pvalues = np.ones(m)
    for rank_idx, orig_idx in enumerate(order):
        raw_p = raw_results[orig_idx][3]
        holm_pvalues[orig_idx] = min(1.0, raw_p * (m - rank_idx))
    # Enforce monotonicity
    for k in range(1, m):
        holm_pvalues[order[k]] = max(holm_pvalues[order[k]], holm_pvalues[order[k - 1]])

    results = []
    for idx_r, (si, sj, z, p_raw, cd) in enumerate(raw_results):
        results.append({
            "state_i": si,
            "state_j": sj,
            "z_stat": z,
            "pvalue_raw": p_raw,
            "pvalue_holm": float(holm_pvalues[idx_r]),
            "significant": bool(holm_pvalues[idx_r] < 0.05),
            "cliffs_delta": cd,
        })
    return results


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size: (concordant - discordant) / (n_x * n_y)."""
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    concordant = np.sum(x[:, None] > y[None, :])
    discordant = np.sum(x[:, None] < y[None, :])
    return float((concordant - discordant) / (len(x) * len(y)))


def _check_regime_separation_pass(tests: dict) -> bool:
    """Return True if Part B pass criteria are satisfied."""
    # sigma4w KW significant + at least one pairwise
    sigma_ok = False
    if "sigma4w" in tests:
        t = tests["sigma4w"]
        kw_sig = t["kw_pvalue"] < 0.05
        any_pairwise = any(p["significant"] for p in t["pairwise"])
        sigma_ok = kw_sig and any_pairwise

    # mdd8w KW significant + at least one pairwise
    mdd_ok = False
    if "mdd8w" in tests:
        t = tests["mdd8w"]
        kw_sig = t["kw_pvalue"] < 0.05
        any_pairwise = any(p["significant"] for p in t["pairwise"])
        mdd_ok = kw_sig and any_pairwise

    # At least one S0/S1 vs S3/S4 pairwise significant on either metric
    low_high_ok = False
    for metric in ["sigma4w", "mdd8w"]:
        if metric not in tests:
            continue
        for p in tests[metric]["pairwise"]:
            si, sj = p["state_i"], p["state_j"]
            if {si, sj} & {0, 1} and {si, sj} & {3, 4} and p["significant"]:
                low_high_ok = True

    return sigma_ok and mdd_ok and low_high_ok


# ---------------------------------------------------------------------------
# Part A — Signal Predictive Power
# ---------------------------------------------------------------------------


def run_signal_predictive(
    df: pd.DataFrame,
    high_conf_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run full Part A signal predictive analysis.

    Returns DataFrame with one row per (signal, horizon, target, subsample).
    """
    records = []
    for signal in SIGNALS:
        for h in HORIZONS:
            for target_base in FORWARD_TARGETS:
                col = f"{target_base}_{h}w"
                if col not in df.columns:
                    continue
                rec_full = _predictive_row(df, signal, h, col, subsample="full")
                records.append(rec_full)
                rec_hc = _predictive_row(high_conf_df, signal, h, col, subsample="high_conf")
                records.append(rec_hc)

    return pd.DataFrame(records)


def _predictive_row(
    df: pd.DataFrame, signal: str, h: int, target_col: str, subsample: str
) -> dict:
    sub = df[[signal, target_col]].dropna()
    x = sub[signal].values
    y = sub[target_col].values
    n = len(x)

    if n < 10:
        return {
            "signal": signal, "horizon_weeks": h, "target": target_col,
            "subsample": subsample, "n_obs": n,
            "spearman_rho": np.nan, "spearman_pvalue": np.nan,
            "ols_beta": np.nan, "hac_tstat": np.nan, "hac_pvalue": np.nan,
            "tercile_bottom_median": np.nan, "tercile_top_median": np.nan,
            "tercile_spread": np.nan,
        }

    sp_rho, sp_pval = stats.spearmanr(x, y)

    lag = max(h - 1, 4)
    beta, tstat, pval = _ols_hac(y, x, lag)

    q33, q67 = np.nanpercentile(x, [33.3, 66.7])
    bot_med = float(np.median(y[x <= q33]))
    top_med = float(np.median(y[x >= q67]))

    return {
        "signal": signal,
        "horizon_weeks": h,
        "target": target_col,
        "subsample": subsample,
        "n_obs": n,
        "spearman_rho": float(sp_rho),
        "spearman_pvalue": float(sp_pval),
        "ols_beta": float(beta),
        "hac_tstat": float(tstat),
        "hac_pvalue": float(pval),
        "tercile_bottom_median": bot_med,
        "tercile_top_median": top_med,
        "tercile_spread": float(top_med - bot_med),
    }


def _ols_hac(y: np.ndarray, x: np.ndarray, lag: int) -> tuple[float, float, float]:
    """OLS with Newey-West Bartlett-kernel HAC standard errors.

    Returns (beta, t_stat, two_sided_pvalue) for the slope coefficient.
    Includes intercept internally but only returns slope stats.
    """
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y
    resids = y - X @ beta_hat

    # Newey-West sandwich estimator: V = (X'X)^{-1} S (X'X)^{-1}
    # where S = sum_{j=0}^{L} w_j * sum_t u_t u_{t-j} X_t X_{t-j}'  (j>0: symmetric)
    S = np.zeros((2, 2))
    for j in range(lag + 1):
        w = 1.0 - j / (lag + 1)  # Bartlett kernel
        gamma = np.zeros((2, 2))
        for t in range(j, n):
            gamma += resids[t] * resids[t - j] * np.outer(X[t], X[t - j])
        if j == 0:
            S += gamma
        else:
            S += w * (gamma + gamma.T)

    V_hac = XtX_inv @ S @ XtX_inv
    se_slope = np.sqrt(max(V_hac[1, 1], 0.0))
    beta_slope = beta_hat[1]

    if se_slope == 0:
        return float(beta_slope), np.nan, np.nan

    t_stat = beta_slope / se_slope
    p_val = 2.0 * stats.norm.sf(abs(t_stat))
    return float(beta_slope), float(t_stat), float(p_val)


def check_signal_predictive_pass(summary_df: pd.DataFrame) -> bool:
    """Return True if Part A pass criteria are satisfied.

    Priority: rho_t > h_t > s_t. Need any one to pass at h=4w or 8w
    on fwd_vol or fwd_mdd in the full subsample, AND direction consistent
    in high_conf subsample.
    """
    # Directional hypotheses: signal -> expected sign of beta for fwd_vol/fwd_mdd
    expected_pos = {"rho_t", "h_t", "s_t"}  # positive relationship with risk targets

    for signal in ["rho_t", "h_t", "s_t"]:
        for h in [4, 8]:
            for target_base in ["fwd_vol", "fwd_mdd"]:
                target_col = f"{target_base}_{h}w"
                full_row = summary_df[
                    (summary_df["signal"] == signal)
                    & (summary_df["horizon_weeks"] == h)
                    & (summary_df["target"] == target_col)
                    & (summary_df["subsample"] == "full")
                ]
                hc_row = summary_df[
                    (summary_df["signal"] == signal)
                    & (summary_df["horizon_weeks"] == h)
                    & (summary_df["target"] == target_col)
                    & (summary_df["subsample"] == "high_conf")
                ]
                if full_row.empty or hc_row.empty:
                    continue

                fr = full_row.iloc[0]
                hr = hc_row.iloc[0]

                direction_correct = (
                    signal in expected_pos and fr["ols_beta"] > 0
                ) or (signal not in expected_pos and fr["ols_beta"] < 0)

                if not direction_correct:
                    continue
                if pd.isna(fr["hac_pvalue"]) or fr["hac_pvalue"] >= 0.10:
                    continue
                if pd.isna(fr["tercile_spread"]) or fr["tercile_spread"] <= 0:
                    continue
                # Direction must not reverse in high-confidence subsample
                if not pd.isna(hr["ols_beta"]) and hr["ols_beta"] * fr["ols_beta"] < 0:
                    continue

                return True

    return False
