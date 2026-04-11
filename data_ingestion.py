import os
import numpy as np
import yfinance as yf
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM",
           "GS", "SPY", "QQQ", "TSLA", "NVDA"]

END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE       = (datetime.today() - timedelta(days=5*365)).strftime("%Y-%m-%d")
FRED_START_DATE  = (datetime.today() - timedelta(days=5*365 + 180)).strftime("%Y-%m-%d")

FRED_SERIES = {
    "cpi"         : "CPIAUCSL",
    "fed_rate"    : "FEDFUNDS",
    "yield_10y"   : "GS10",
    "yield_2y"    : "GS2",
    "unemployment": "UNRATE",
}

# ── 1. Market Data ────────────────────────────────────────────────────────────
def fetch_market_data(tickers, start, end):
    print(f"\n[1/3] Fetching market data for {len(tickers)} tickers...")

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    df  = raw.stack(level=1).rename_axis(["Date", "Ticker"]).reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["daily_return"] = df.groupby("ticker")["close"].pct_change()
    df["log_return"]   = df.groupby("ticker")["close"].transform(
                             lambda x: np.log(x / x.shift(1)))

    print(f"    Shape: {df.shape} | {df['date'].min()} → {df['date'].max()}")
    return df

# ── 2. Macro Data ─────────────────────────────────────────────────────────────
def fetch_macro_data(series_dict, start, end):
    print(f"\n[2/3] Fetching {len(series_dict)} FRED macro series...")
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY missing from .env")

    fred   = Fred(api_key=api_key)
    frames = {}

    for name, sid in series_dict.items():
        try:
            # Fetch extra 6 months back so merge_asof always has a prior value
            s = fred.get_series(sid)
            s = s[s.index <= end]          # only trim the end, not the start
            frames[name] = s
            print(f"    {name:15s} ({sid}): {len(s)} obs | "
                  f"{s.index.min().date()} → {s.index.max().date()}")
        except Exception as e:
            print(f"    WARN: {name} failed — {e}")

    if not frames:
        raise RuntimeError("No FRED data fetched — check API key")

    macro_df = pd.DataFrame(frames)
    macro_df.index = pd.to_datetime(macro_df.index)

    # Forward-fill to every business day — starting from FRED_START_DATE
    bdays    = pd.date_range(start=FRED_START_DATE, end=end, freq="B")
    macro_df = macro_df.reindex(bdays).ffill().dropna()  # drop rows still NaN after ffill
    macro_df.index.name = "date"
    macro_df = macro_df.reset_index()

    macro_df["yield_spread"] = macro_df["yield_10y"] - macro_df["yield_2y"]
    macro_df["cpi_mom"]      = macro_df["cpi"].pct_change()

    print(f"    Macro shape after forward-fill: {macro_df.shape}")
    return macro_df

# ── 3. Merge & Save ───────────────────────────────────────────────────────────
def merge_and_save(market_df, macro_df):
    print(f"\n[3/3] Merging and saving to /data ...")

    market_df["date"] = pd.to_datetime(market_df["date"])
    macro_df["date"]  = pd.to_datetime(macro_df["date"])

    merged = pd.merge(market_df, macro_df, on="date", how="left")
    merged = merged.dropna(subset=["close"])

    market_df.to_parquet("data/market_data.parquet", index=False)
    macro_df.to_parquet("data/macro_data.parquet",   index=False)
    merged.to_parquet("data/merged_data.parquet",    index=False)

    print(f"    market_data.parquet : {market_df.shape}")
    print(f"    macro_data.parquet  : {macro_df.shape}")
    print(f"    merged_data.parquet : {merged.shape}")
    return merged

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  AlphaLens — Day 1: Data Ingestion")
    print("=" * 50)

    market_df = fetch_market_data(TICKERS, START_DATE, END_DATE)
    macro_df  = fetch_macro_data(FRED_SERIES, START_DATE, END_DATE)
    merged_df = merge_and_save(market_df, macro_df)

    print(f"\nDone. {len(merged_df)} rows, {len(merged_df.columns)} columns.")
    print(f"Columns: {list(merged_df.columns)}")
    print("\nDay 1 complete — data ready for feature engineering!")
