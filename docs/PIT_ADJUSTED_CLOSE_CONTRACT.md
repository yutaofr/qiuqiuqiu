# PIT Adjusted Close Contract

## Purpose

The micro layer may use adjusted prices only when those prices are adjusted with information knowable at the decision time. Hindsight-adjusted vendor close series are invalid for rolling windows because they rewrite past prices using future corporate actions.

## Required Fields

- `trade_date`: the market date of the raw close.
- `asof_timestamp`: the timestamp at which the raw close and corporate-action factors are knowable.
- `raw_close`: the unadjusted traded close for `trade_date`.
- `split_factor_cum_pti`: cumulative split factor knowable at `asof_timestamp`.
- `dividend_factor_cum_pti`: cumulative dividend factor knowable at `asof_timestamp`.
- `adj_close_pti`: `raw_close * split_factor_cum_pti * dividend_factor_cum_pti`.

## Knowable At Time t

Data is knowable at time `t` only if its source timestamp is less than or equal to the decision timestamp. A weekly decision may not use corporate actions, membership changes, vendor restatements, or adjusted-price factors published after that week’s decision timestamp.

## Allowed Sources

- Raw closes with exchange trade dates and publication timestamps.
- Split and dividend action records with effective dates and source/as-of timestamps.
- Internally archived daily PIT price bars and corporate-action factors.

## Forbidden Sources

- Hindsight-adjusted close series that rewrite historical prices using future corporate actions.
- Vendor total-return or adjusted-close histories without as-of factor history.
- Any price window whose adjustment basis was not knowable on or before the requested `asof_timestamp`.

## Engine Interface

`PITAdjustmentEngine.get_adj_close(ticker, trade_date, asof)` returns one PIT adjusted close.

`PITAdjustmentEngine.get_adjusted_window(ticker, end_date, window, asof)` returns a window ending at `end_date`, with every raw close adjusted to the corporate-action basis knowable at `asof`:

```text
P_adj(tau | asof) = P_raw(tau) * CUM_FAC(asof) / CUM_FAC(tau)
CUM_FAC(d) = split_factor_cum(d) * dividend_factor_cum(d)
```

## Failure Modes

- `HindsightAdjustedDataError`: only hindsight-adjusted prices are available.
- `DataNotAvailableError`: no PIT source is wired.
- `InsufficientHistoryError`: fewer than the requested number of PIT observations are available.

## Degraded Mode

If PIT adjusted prices cannot be guaranteed, the micro layer must not produce `h_t` and the risk layer must not produce production `rho_t`. The system remains in lightweight state/stress replay mode and reports `h_t = None`, `rho_t = None`.
