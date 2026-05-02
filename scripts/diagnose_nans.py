import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
from qqq_cycle.core.state_layer import compute_state_layer

STAGING_CSV = Path("cache/real_replay/staging/weekly_inputs.csv")

def diagnose():
    if not STAGING_CSV.exists():
        print(f"Error: {STAGING_CSV} not found")
        return
    
    df = pd.read_csv(STAGING_CSV, index_col=0, parse_dates=True)
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    print("\nNull counts per column:")
    print(df.isnull().sum())
    
    state = compute_state_layer(df)
    theta = state[["H", "I"]]
    finite = theta.dropna()
    
    print(f"\nFinite theta rows: {len(finite)}")
    if not finite.empty:
        print(f"Finite range: {finite.index[0]} to {finite.index[-1]}")
    else:
        # Check where each factor becomes finite
        for col in ["L", "T", "P", "E", "H", "I"]:
            f = state[col].dropna()
            if not f.empty:
                print(f"{col} becomes finite at {f.index[0]} (count: {len(f)})")
            else:
                print(f"{col} is ALL NULL")

if __name__ == "__main__":
    diagnose()
