import pandas as pd
import yfinance as yf
from pathlib import Path
import sys
import warnings
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

CACHE_DIR = Path("cache/micro")
PRICES_DIR = CACHE_DIR / "prices"
HOLDINGS_FILE = Path("normalized/qqq_holdings_2026-05-01_normalized.csv")

def fetch_prices(ticker: str, start: str, end: str):
    print(f"  Fetching {ticker}...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use a slightly longer window to ensure we have enough history for the 60-day window
            df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        
        if df is None or df.empty:
            print(f"    Warning: No data for {ticker}")
            return None
            
        # Handle multi-index columns if yfinance returns them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Ensure required columns exist
        if "Close" not in df.columns or "Adj Close" not in df.columns:
            print(f"    Warning: Missing required columns for {ticker}")
            return None

        # Create the format expected by CsvPITAdjustmentEngine:
        # trade_date, raw_close, adj_close, asof_timestamp
        result = pd.DataFrame({
            "trade_date": df.index.normalize(),
            "raw_close": df["Close"].values,
            "adj_close": df["Adj Close"].values,
            "asof_timestamp": (df.index.normalize() + pd.Timedelta(hours=16)).strftime("%Y-%m-%dT%H:%M:%S")
        })
        return result.dropna()
    except Exception as e:
        print(f"    Error fetching {ticker}: {e}")
        return None

def main():
    if not HOLDINGS_FILE.exists():
        print(f"Error: {HOLDINGS_FILE} not found")
        return

    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    holdings = pd.read_csv(HOLDINGS_FILE)
    
    # We need to map canonical_symbol (for yfinance) to instrument_id (for storage)
    mapping = holdings[["canonical_symbol", "instrument_id", "normalized_weight"]].dropna()
    
    print(f"Found {len(mapping)} tickers in holdings for 2026-05-01")
    
    # We need at least 156 weeks of history for z_wrob_156
    start_date = "2020-01-01"
    end_date = "2026-05-03"
    
    successful_instrument_ids = []
    
    for _, row in mapping.iterrows():
        symbol = row["canonical_symbol"]
        instrument_id = row["instrument_id"]
        
        df = fetch_prices(symbol, start_date, end_date)
        if df is not None and len(df) >= 60:
            # Use instrument_id (e.g. CUSIP:...) as the filename
            # Note: need to handle colon in filename if it exists
            path = PRICES_DIR / f"{instrument_id}.csv"
            df.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S")
            successful_instrument_ids.append(instrument_id)
        else:
            print(f"    Skipped {symbol} / {instrument_id} (insufficient data or error)")

    print(f"\nSuccessfully updated {len(successful_instrument_ids)} tickers.")
    
    # Update constituents.csv and weights.csv for 2026-05-01
    trade_date = "2026-05-01"
    asof = "2026-05-02T12:00:00" # After the trade date
    
    # Load existing or create new
    if (CACHE_DIR / "constituents.csv").exists():
        cons = pd.read_csv(CACHE_DIR / "constituents.csv")
    else:
        cons = pd.DataFrame(columns=["trade_date", "ticker", "asof_timestamp"])
        
    if (CACHE_DIR / "weights.csv").exists():
        weights = pd.read_csv(CACHE_DIR / "weights.csv")
    else:
        weights = pd.DataFrame(columns=["trade_date", "ticker", "weight", "asof_timestamp"])

    # Remove existing 2026-05-01 if any
    cons = cons[cons["trade_date"] != trade_date]
    weights = weights[weights["trade_date"] != trade_date]
    
    new_cons = []
    new_weights = []
    
    # Only include tickers that we successfully fetched prices for
    for _, row in mapping.iterrows():
        instrument_id = row["instrument_id"]
        if instrument_id in successful_instrument_ids:
            new_cons.append({"trade_date": trade_date, "ticker": instrument_id, "asof_timestamp": asof})
            new_weights.append({
                "trade_date": trade_date, 
                "ticker": instrument_id, 
                "weight": row["normalized_weight"], 
                "asof_timestamp": asof
            })
            
    cons = pd.concat([cons, pd.DataFrame(new_cons)], ignore_index=True)
    weights = pd.concat([weights, pd.DataFrame(new_weights)], ignore_index=True)
    
    cons.to_csv(CACHE_DIR / "constituents.csv", index=False)
    weights.to_csv(CACHE_DIR / "weights.csv", index=False)
    
    print(f"Updated constituents.csv and weights.csv for {trade_date}")

if __name__ == "__main__":
    main()
