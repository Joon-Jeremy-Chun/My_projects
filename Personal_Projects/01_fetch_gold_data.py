# code_01_fetch_gold_data.py
# Fetch gold price data and save to CSV
# Saves: ./data/gold_prices.csv (by default)

# import sys
# import subprocess

# def install_and_import(package):
#     try:
#         __import__(package)
#         print(f"'{package}' is already installed.")
#     except ImportError:
#         print(f"'{package}' not found. Installing...")
#         subprocess.check_call([sys.executable, "-m", "pip", "install", package])
#         __import__(package)
#         print(f"'{package}' installed successfully.")

# install_and_import("yfinance")

# import yfinance as yf
# print("yfinance version:", yf.__version__)




#%%


import argparse
from pathlib import Path
import sys
import pandas as pd




def fetch_with_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance") from e

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    if df is None or df.empty:
        raise RuntimeError(f"No data returned from yfinance for ticker={ticker}.")

    # ✅ IMPORTANT: flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Move index date to a column
    df = df.reset_index()

    # Standardize date column name
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    # ✅ Choose price column robustly
    if "Adj Close" in df.columns and df["Adj Close"].notna().any():
        df["Price"] = df["Adj Close"]
    elif "Close" in df.columns and df["Close"].notna().any():
        df["Price"] = df["Close"]
    else:
        raise RuntimeError(
            f"Could not find usable price column. Columns available: {list(df.columns)}"
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "Price"]].dropna(subset=["Price"]).sort_values("Date").reset_index(drop=True)
    return df



def main():
    parser = argparse.ArgumentParser(description="Fetch gold price data and save to CSV.")
    parser.add_argument("--ticker", type=str, default="GC=F",
                        help="Ticker symbol (default: GC=F for Gold futures on Yahoo Finance). "
                             "Alternative: XAUUSD=X (spot-ish) depending on availability.")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2026-01-01", help="End date YYYY-MM-DD")
    parser.add_argument("--out", type=str, default="./data/gold_prices.csv", help="Output CSV path")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = fetch_with_yfinance(args.ticker, args.start, args.end)
    except Exception as e:
        print(f"[ERROR] Failed to fetch data: {e}", file=sys.stderr)
        sys.exit(1)

    df.to_csv(out_path, index=False)
    print(f"[OK] Saved {len(df):,} rows to: {out_path.resolve()}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
