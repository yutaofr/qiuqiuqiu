#!/usr/bin/env python3
"""Capture QQQ daily prices from yfinance for diagnostic macro replay."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "cache" / "real_replay" / "raw" / "qqq_macro_captured.csv"

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2000-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date (default: today)")
    args = parser.parse_args()

    end = args.end or datetime.now().strftime("%Y-%m-%d")
    print(f"Fetching QQQ prices from {args.start} to {end}...")

    try:
        df = yf.download("QQQ", start=args.start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            print("ERROR: yfinance returned empty data")
            sys.exit(1)
        
        # Handle yfinance MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.get_level_values(0):
                close_series = df["Close"].iloc[:, 0]
            else:
                print(f"ERROR: 'Close' column not found in {df.columns}")
                sys.exit(1)
        else:
            close_series = df["Close"]

        # Format for MacroMarketPriceContract
        # Required columns: trade_date, ticker, close, source_name, fetch_timestamp, price_basis
        fetch_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        out = pd.DataFrame({
            "trade_date": close_series.index.strftime("%Y-%m-%d"),
            "ticker": "QQQ",
            "close": close_series.values.flatten(),
            "source_name": "yfinance",
            "fetch_timestamp": fetch_ts,
            "price_basis": "vendor_backward_adjusted"
        })
        
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUTPUT_PATH, index=False)
        print(f"Successfully captured {len(out)} rows to {OUTPUT_PATH}")
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()
