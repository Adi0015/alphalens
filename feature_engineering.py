import numpy as np
import pandas as pd
import ta


def add_technical_indicators(df):
    print("\n[1/3] Computing technical indicators...")
    frames = []
    spy_close = df[df["ticker"] == "SPY"].set_index("date")["close"]

    for ticker, group in df.groupby("ticker"):
        g     = group.sort_values("date").copy()
        close = g["close"]
        high  = g["high"]
        low   = g["low"]
        vol   = g["volume"]

        # ── Trend ─────────────────────────────────────────────
        g["ema_12"]      = ta.trend.EMAIndicator(close, window=12).ema_indicator()
        g["ema_26"]      = ta.trend.EMAIndicator(close, window=26).ema_indicator()
        g["macd"]        = ta.trend.MACD(close).macd()
        g["macd_signal"] = ta.trend.MACD(close).macd_signal()
        g["macd_diff"]   = ta.trend.MACD(close).macd_diff()

        # ── Momentum ──────────────────────────────────────────
        g["rsi"]         = ta.momentum.RSIIndicator(close, window=14).rsi()
        g["roc_10"]      = ta.momentum.ROCIndicator(close, window=10).roc()
        g["williams_r"]  = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()

        # ── Volatility ────────────────────────────────────────
        bb               = ta.volatility.BollingerBands(close, window=20)
        g["bb_upper"]    = bb.bollinger_hband()
        g["bb_lower"]    = bb.bollinger_lband()
        g["bb_width"]    = bb.bollinger_wband()
        g["bb_pct"]      = bb.bollinger_pband()
        g["atr"]         = ta.volatility.AverageTrueRange(
                               high, low, close).average_true_range()

        # ── Volume ────────────────────────────────────────────
        g["obv"]         = ta.volume.OnBalanceVolumeIndicator(
                               close, vol).on_balance_volume()
        g["vol_ma20"]    = vol.rolling(20).mean()
        g["vol_ratio"]   = vol / g["vol_ma20"]

        # ── Price derived ─────────────────────────────────────
        g["return_5d"]      = close.pct_change(5)
        g["return_20d"]     = close.pct_change(20)
        g["dist_52w_high"]  = close / close.rolling(252).max() - 1
        g["dist_52w_low"]   = close / close.rolling(252).min() - 1

        # ── NEW: Realized volatility ──────────────────────────
        g["realized_vol_20"] = (
            g["daily_return"].rolling(20).std() * np.sqrt(252)
        )

        # ── NEW: Z-score mean reversion signal ────────────────
        g["zscore_20"] = (
            (close - close.rolling(20).mean()) /
            close.rolling(20).std()
        )

        # ── NEW: Overnight gap ────────────────────────────────
        g["overnight_gap"] = (
            g["open"] - g["close"].shift(1)
        ) / g["close"].shift(1)

        # ── NEW: Price vs SPY (relative strength) ─────────────
        g["date_idx"] = g["date"]
        spy_aligned   = g["date_idx"].map(spy_close)
        if ticker != "SPY" and spy_aligned.notna().any():
            g["price_vs_spy"] = close.values / spy_aligned.values - 1
        else:
            g["price_vs_spy"] = 0.0

        # ── NEW: Higher timeframe trend ───────────────────────
        g["ema_50"]         = ta.trend.EMAIndicator(close, window=50).ema_indicator()
        g["ema_200"]        = ta.trend.EMAIndicator(close, window=200).ema_indicator()
        g["above_ema50"]    = (close > g["ema_50"]).astype(int)
        g["above_ema200"]   = (close > g["ema_200"]).astype(int)
        g["ema_50_200_cross"] = (g["ema_50"] > g["ema_200"]).astype(int)

        # ── NEW: RSI divergence ───────────────────────────────
        g["rsi_ma10"]      = g["rsi"].rolling(10).mean()
        g["rsi_divergence"]= g["rsi"] - g["rsi_ma10"]

        # ── NEW: Volume spike ─────────────────────────────────
        g["vol_spike"]     = (g["vol_ratio"] > 2.0).astype(int)

        frames.append(g)

    result = (pd.concat(frames)
                .sort_values(["ticker", "date"])
                .reset_index(drop=True)
                .drop(columns=["date_idx"], errors="ignore"))

    print(f"    Done. Shape: {result.shape}")
    return result


def add_target(df):
    print("\n[2/3] Adding target variable...")
    frames = []
    for ticker, group in df.groupby("ticker"):
        g = group.sort_values("date").copy()
        g["future_return_5d"] = g["close"].pct_change(5).shift(-5)
        g["target"]           = (g["future_return_5d"] > 0).astype(int)
        frames.append(g)
    result = (pd.concat(frames)
                .sort_values(["ticker", "date"])
                .reset_index(drop=True))
    print(f"    Target distribution:\n{result['target'].value_counts()}")
    return result


def add_interaction_features(df):
    """Cross-feature interactions that help tree models find non-linear patterns."""
    print("\n[3/3] Adding interaction features...")

    # RSI × volume spike — oversold + high volume = strong reversal signal
    df["rsi_vol_spike"]      = df["rsi"] * df["vol_spike"]

    # MACD × sentiment — technical + sentiment alignment
    df["macd_sentiment"]     = df["macd_diff"] * df["sentiment_mean"]

    # Volatility × momentum — momentum in high-vol environment
    df["vol_momentum"]       = df["realized_vol_20"] * df["return_5d"]

    # Z-score × RSI — double mean-reversion confirmation
    df["zscore_rsi"]         = df["zscore_20"] * (df["rsi"] - 50)

    # Yield spread × sentiment — macro + news alignment
    df["macro_sentiment"]    = df["yield_spread"] * df["sentiment_mean"]

    print(f"    Done. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    print("=" * 50)
    print("  AlphaLens — Day 2a: Feature Engineering (Enhanced)")
    print("=" * 50)

    df = pd.read_parquet("data/merged_data.parquet")
    print(f"Loaded: {df.shape}")

    df = add_technical_indicators(df)
    df = add_target(df)

    # Load and merge sentiment before interactions
    try:
        sentiment = pd.read_parquet("data/sentiment_raw.parquet")
        sentiment["date"] = pd.to_datetime(sentiment["date"]).dt.normalize()
        df["date"]        = pd.to_datetime(df["date"]).dt.normalize()
        df = pd.merge(df, sentiment, on=["date", "ticker"], how="left")
        for col in ["sentiment_mean", "sentiment_std", "headline_count"]:
            df[col] = df.groupby("ticker")[col].ffill().fillna(0)
        print(f"\nSentiment merged. Shape: {df.shape}")
    except FileNotFoundError:
        print("\nWARN: sentiment_raw.parquet not found — run sentiment_engine.py first")
        df["sentiment_mean"]  = 0.0
        df["sentiment_std"]   = 0.0
        df["headline_count"]  = 0.0

    df = add_interaction_features(df)

    df.to_parquet("data/features_data.parquet",  index=False)
    df.to_parquet("data/final_features.parquet", index=False)
    print(f"\nSaved: data/final_features.parquet {df.shape}")
    print(f"Total features: {len(df.columns)}")
