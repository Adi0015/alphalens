import numpy as np
import pandas as pd
import ta

def add_technical_indicators(df):
    print("\n[1/2] Computing technical indicators...")
    frames = []

    for ticker, group in df.groupby("ticker"):
        g = group.sort_values("date").copy()
        close = g["close"]
        high  = g["high"]
        low   = g["low"]
        vol   = g["volume"]

        # ── Trend ────────────────────────────────────────────
        g["ema_12"]      = ta.trend.EMAIndicator(close, window=12).ema_indicator()
        g["ema_26"]      = ta.trend.EMAIndicator(close, window=26).ema_indicator()
        g["macd"]        = ta.trend.MACD(close).macd()
        g["macd_signal"] = ta.trend.MACD(close).macd_signal()
        g["macd_diff"]   = ta.trend.MACD(close).macd_diff()

        # ── Momentum ─────────────────────────────────────────
        g["rsi"]         = ta.momentum.RSIIndicator(close, window=14).rsi()
        g["roc_10"]      = ta.momentum.ROCIndicator(close, window=10).roc()
        g["williams_r"]  = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()

        # ── Volatility ────────────────────────────────────────
        bb               = ta.volatility.BollingerBands(close, window=20)
        g["bb_upper"]    = bb.bollinger_hband()
        g["bb_lower"]    = bb.bollinger_lband()
        g["bb_width"]    = bb.bollinger_wband()
        g["bb_pct"]      = bb.bollinger_pband()
        g["atr"]         = ta.volatility.AverageTrueRange(high, low, close).average_true_range()

        # ── Volume ────────────────────────────────────────────
        g["obv"]         = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
        g["vol_ma20"]    = vol.rolling(20).mean()
        g["vol_ratio"]   = vol / g["vol_ma20"]

        # ── Price derived ─────────────────────────────────────
        g["return_5d"]     = close.pct_change(5)
        g["return_20d"]    = close.pct_change(20)
        g["dist_52w_high"] = close / close.rolling(252).max() - 1
        g["dist_52w_low"]  = close / close.rolling(252).min() - 1

        frames.append(g)

    result = pd.concat(frames).sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"    Done. Shape: {result.shape}")
    return result


def add_target(df):
    print("\n[2/2] Adding target variable...")
    frames = []
    for ticker, group in df.groupby("ticker"):
        g = group.sort_values("date").copy()
        g["future_return_5d"] = g["close"].pct_change(5).shift(-5)
        g["target"]           = (g["future_return_5d"] > 0).astype(int)
        frames.append(g)
    result = pd.concat(frames).sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"    Target distribution:\n{result['target'].value_counts()}")
    return result


if __name__ == "__main__":
    print("=" * 50)
    print("  AlphaLens — Day 2a: Feature Engineering")
    print("=" * 50)

    df = pd.read_parquet("data/merged_data.parquet")
    print(f"Loaded merged data: {df.shape}")

    df = add_technical_indicators(df)
    df = add_target(df)

    df.to_parquet("data/features_data.parquet", index=False)
    print(f"\nSaved: data/features_data.parquet {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nRun sentiment_engine.py next!")
