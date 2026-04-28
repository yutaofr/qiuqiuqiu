"""Fetch and seed real QQQ micro data: constituents, weights, per-ticker prices.

Usage:
    python scripts/seed_micro_data.py [--start 2021-01-01] [--end 2024-12-31]

Outputs:
    cache/micro/constituents.csv
    cache/micro/weights.csv
    cache/micro/prices/{ticker}.csv (one file per ticker)
    cache/micro/seed_manifest.json

PIT convention:
    asof_timestamp = trade_date (data is available end-of-day on trade_date).
    yfinance applies split/dividend adjustments retroactively at fetch time.
    This is a known limitation documented in seed_manifest.json and is acceptable
    for Phase 7 backtesting (structural routing proof, not live production).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path("cache/micro")
PRICES_DIR = CACHE_DIR / "prices"

# Top QQQ constituents with approximate 2024 weights (normalized within this set).
# These stocks have been continuously present in QQQ throughout 2021-2024.
# Weights are illustrative; the micro layer uses smoothed weights so small
# inaccuracies do not materially affect the routing gate.
_TOP_QQQ_HOLDINGS: dict[str, float] = {
    "AAPL":  9.0,
    "MSFT":  8.0,
    "NVDA":  8.0,
    "AMZN":  5.0,
    "META":  5.0,
    "GOOGL": 2.5,
    "GOOG":  2.5,
    "TSLA":  3.0,
    "AVGO":  3.0,
    "COST":  2.0,
    "NFLX":  1.5,
    "AMD":   1.5,
    "ADBE":  1.0,
    "QCOM":  2.0,
    "PEP":   1.5,
    "INTU":  2.0,
    "CSCO":  1.5,
    "TXN":   1.5,
    "AMGN":  1.0,
    "AMAT":  1.0,
}

# Normalize weights so they sum to 1.0.
_TOTAL_WEIGHT = sum(_TOP_QQQ_HOLDINGS.values())
_WEIGHTS: dict[str, float] = {k: v / _TOTAL_WEIGHT for k, v in _TOP_QQQ_HOLDINGS.items()}


def _fetch_prices(
    ticker: str, start: str, end: str
) -> pd.DataFrame | None:
    """Download OHLC for a single ticker; return DataFrame or None on failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )
        if df is None or df.empty:
            print(f"  SKIP {ticker}: empty download", file=sys.stderr)
            return None
        # yfinance returns MultiIndex columns when auto_adjust=False
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        required = {"Close", "Adj Close"}
        missing = required - set(df.columns)
        if missing:
            print(f"  SKIP {ticker}: missing columns {missing}", file=sys.stderr)
            return None
        result = pd.DataFrame({
            "trade_date": df.index.normalize(),
            "raw_close": df["Close"].values,
            "adj_close": df["Adj Close"].values,
            "asof_timestamp": df.index.normalize(),
        })
        result = result.dropna(subset=["raw_close", "adj_close"])
        result = result[result["raw_close"] > 0]
        if len(result) < 60:
            print(f"  SKIP {ticker}: only {len(result)} rows (< 60)", file=sys.stderr)
            return None
        return result
    except Exception as exc:
        print(f"  SKIP {ticker}: {exc}", file=sys.stderr)
        return None


def _write_prices(ticker: str, df: pd.DataFrame) -> None:
    path = PRICES_DIR / f"{ticker}.csv"
    df.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S")


def _write_constituents(tickers: list[str], trading_days: pd.DatetimeIndex) -> None:
    """Write one row per (trade_date, ticker) for each trading day."""
    rows = [
        {"trade_date": d.strftime("%Y-%m-%d"), "ticker": t, "asof_timestamp": d.strftime("%Y-%m-%dT16:00:00")}
        for d in trading_days
        for t in tickers
    ]
    df = pd.DataFrame(rows)
    df.to_csv(CACHE_DIR / "constituents.csv", index=False)
    print(f"  wrote {len(df)} constituent rows for {len(tickers)} tickers over {len(trading_days)} days")


def _write_weights(tickers: list[str], trading_days: pd.DatetimeIndex) -> None:
    """Write one weight row per (trade_date, ticker) for each trading day."""
    rows = [
        {
            "trade_date": d.strftime("%Y-%m-%d"),
            "ticker": t,
            "weight": round(_WEIGHTS[t], 6),
            "asof_timestamp": d.strftime("%Y-%m-%dT16:00:00"),
        }
        for d in trading_days
        for t in tickers
    ]
    df = pd.DataFrame(rows)
    df.to_csv(CACHE_DIR / "weights.csv", index=False)
    print(f"  wrote {len(df)} weight rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed QQQ micro data for Phase 7")
    parser.add_argument("--start", default="2021-01-01", help="Start date (inclusive)")
    parser.add_argument("--end", default="2024-12-31", help="End date (inclusive)")
    args = parser.parse_args()

    PRICES_DIR.mkdir(parents=True, exist_ok=True)

    fetch_timestamp = datetime.now(timezone.utc).isoformat()
    print(f"Seeding QQQ micro data: {args.start} to {args.end}")
    print(f"Fetch timestamp: {fetch_timestamp}")

    # Download prices for all tickers.
    successful: list[str] = []
    skipped: list[str] = []
    trading_days_union: pd.DatetimeIndex = pd.DatetimeIndex([])

    for ticker in _TOP_QQQ_HOLDINGS:
        print(f"  Fetching {ticker}...")
        df = _fetch_prices(ticker, args.start, args.end)
        if df is None:
            skipped.append(ticker)
            continue
        _write_prices(ticker, df)
        days = pd.DatetimeIndex(df["trade_date"].unique())
        trading_days_union = trading_days_union.union(days)
        successful.append(ticker)
        print(f"  OK {ticker}: {len(df)} rows")

    if len(successful) < 10:
        print(
            f"ERROR: only {len(successful)} tickers succeeded (need >= 10). "
            "Check network connectivity.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Use the intersection of trading days across successful tickers so that
    # constituent/weight CSVs only cover days where at least one price file exists.
    trading_days = trading_days_union.sort_values()
    print(f"\nSuccessful tickers ({len(successful)}): {successful}")
    print(f"Skipped tickers ({len(skipped)}): {skipped}")
    print(f"Trading days covered: {len(trading_days)} ({trading_days[0].date()} to {trading_days[-1].date()})")

    print("\nWriting constituents.csv...")
    _write_constituents(successful, trading_days)

    print("Writing weights.csv...")
    _write_weights(successful, trading_days)

    # Write manifest.
    manifest = {
        "fetch_timestamp": fetch_timestamp,
        "start": args.start,
        "end": args.end,
        "tickers_attempted": list(_TOP_QQQ_HOLDINGS.keys()),
        "tickers_successful": successful,
        "tickers_skipped": skipped,
        "trading_days": len(trading_days),
        "pit_convention": (
            "asof_timestamp = trade_date (EOD). "
            "yfinance applies split/dividend adjustments retroactively at fetch time. "
            "This is documented as a Phase 7 backtesting simplification, not for live production."
        ),
        "weight_source": (
            "Approximate 2024 QQQ top-holding weights, normalized within the seeded ticker set. "
            "Not scraped from Invesco — intended for structural routing proof only."
        ),
    }
    manifest_path = CACHE_DIR / "seed_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()
