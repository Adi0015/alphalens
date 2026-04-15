import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Config ────────────────────────────────────────────────────────────────────

INITIAL_CAPITAL    = 100_000     # $100k starting portfolio
MIN_CONFIDENCE     = 0.58        # only trade high-confidence signals
TRANSACTION_COST   = 0.001       # 0.1% per trade (realistic for ETFs)
TRADING_DAYS       = 252
RISK_FREE_RATE     = 0.05


# ── Load data ─────────────────────────────────────────────────────────────────

def load_backtest_data():
    """Load features, portfolio weights, and ML model."""
    print("\n[1/5] Loading backtest data...")

    df       = pd.read_parquet("data/final_features.parquet")
    df["date"]= pd.to_datetime(df["date"])
    weights  = pd.read_csv("data/portfolio_weights.csv")
    model    = joblib.load("models/xgboost_tuned.pkl")

    print(f"    Features shape  : {df.shape}")
    print(f"    Tickers in portfolio: {weights['ticker'].tolist()}")
    return df, weights, model


# ── Generate signals ──────────────────────────────────────────────────────────

def generate_signals(df, model, weights_df):
    """
    For each date, generate a signal for each ticker.
    Signal = 1 if:
      - regime is bull
      - model confidence >= MIN_CONFIDENCE
      - ticker is in our optimized universe
    """
    print("\n[2/5] Generating trading signals...")

    FEATURE_COLS = [
        "ema_12","ema_26","macd","macd_signal","macd_diff",
        "ema_50","ema_200","above_ema50","above_ema200","ema_50_200_cross",
        "rsi","roc_10","williams_r","rsi_ma10","rsi_divergence",
        "bb_pct","bb_width","atr","realized_vol_20","zscore_20",
        "obv","vol_ratio","vol_spike",
        "return_5d","return_20d","dist_52w_high","dist_52w_low",
        "overnight_gap","price_vs_spy","daily_return","log_return",
        "cpi","fed_rate","yield_10y","yield_2y",
        "yield_spread","cpi_mom","unemployment",
        "sentiment_mean","sentiment_std","headline_count",
        "rsi_vol_spike","macd_sentiment","vol_momentum",
        "zscore_rsi","macro_sentiment",
    ]

    available    = [c for c in FEATURE_COLS if c in df.columns]
    universe     = weights_df["ticker"].tolist()
    df_universe  = df[df["ticker"].isin(universe)].copy()
    df_universe  = df_universe.dropna(subset=available)

    X            = df_universe[available]
    proba        = model.predict_proba(X)[:, 1]

    df_universe["bull_proba"] = proba
    df_universe["ml_signal"]  = (proba >= MIN_CONFIDENCE).astype(int)
    df_universe["regime_bull"]= (df_universe["regime"] == "bull").astype(int)
    df_universe["signal"]     = (
        df_universe["ml_signal"] & df_universe["regime_bull"]
    ).astype(int)

    total_signals = df_universe["signal"].sum()
    active_days   = df_universe.groupby("date")["signal"].max().sum()
    print(f"    Total signals   : {total_signals}")
    print(f"    Active days     : {active_days}")
    print(f"    Signal rate     : {df_universe['signal'].mean():.1%}")

    return df_universe


# ── Run backtest ──────────────────────────────────────────────────────────────

def run_backtest(signals_df, weights_df):
    """
    Vectorized backtest:
    - On signal days, hold the MVO-weighted portfolio
    - Otherwise hold cash (earning risk-free rate)
    - Apply transaction costs on position changes
    """
    print("\n[3/5] Running vectorized backtest...")

    weight_map  = weights_df.set_index("ticker")["bl_weight"].to_dict()
    dates       = sorted(signals_df["date"].unique())
    portfolio   = INITIAL_CAPITAL
    daily_values= []
    prev_invested = False

    for date in dates:
        day_data  = signals_df[signals_df["date"] == date]
        any_signal= day_data["signal"].max() == 1

        if any_signal:
            # Weighted portfolio return for this day
            port_return = 0.0
            for _, row in day_data.iterrows():
                w = weight_map.get(row["ticker"], 0)
                port_return += w * row["daily_return"]

            # Transaction cost on entry
            if not prev_invested:
                portfolio *= (1 - TRANSACTION_COST)

            portfolio    *= (1 + port_return)
            prev_invested = True
        else:
            # Hold cash at daily risk-free rate
            portfolio    *= (1 + RISK_FREE_RATE / TRADING_DAYS)
            if prev_invested:
                portfolio *= (1 - TRANSACTION_COST)   # exit cost
            prev_invested = False

        daily_values.append({
            "date"     : date,
            "portfolio": portfolio,
            "invested" : any_signal,
        })

    equity = pd.DataFrame(daily_values)
    equity["daily_return"] = equity["portfolio"].pct_change().fillna(0)
    print(f"    Final portfolio : ${equity['portfolio'].iloc[-1]:,.2f}")
    print(f"    Total return    : "
          f"{(equity['portfolio'].iloc[-1]/INITIAL_CAPITAL - 1):.2%}")
    return equity


# ── SPY benchmark ──────────────────────────────────────────────────────────────

def get_spy_benchmark(signals_df):
    """Buy-and-hold SPY over the same period."""
    print("\n[4/5] Computing SPY benchmark...")

    spy = (signals_df[signals_df["ticker"] == "SPY"]
             .sort_values("date")[["date", "daily_return"]]
             .drop_duplicates("date")
             .reset_index(drop=True))

    if len(spy) == 0:
        # SPY not in signals_df universe — load directly from features
        df  = pd.read_parquet("data/final_features.parquet")
        spy = (df[df["ticker"] == "SPY"]
                 .sort_values("date")[["date", "daily_return"]]
                 .drop_duplicates("date")
                 .dropna()
                 .reset_index(drop=True))

    portfolio = INITIAL_CAPITAL
    values    = []
    for _, row in spy.iterrows():
        portfolio *= (1 + row["daily_return"])
        values.append({"date": row["date"], "spy_value": portfolio})

    spy_eq  = pd.DataFrame(values).rename(columns={"spy_value": "spy"})
    spy_ret = (spy_eq["spy"].iloc[-1] / INITIAL_CAPITAL - 1)
    print(f"    SPY final value : ${spy_eq['spy'].iloc[-1]:,.2f}")
    print(f"    SPY total return: {spy_ret:.2%}")
    return spy_eq


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(equity, spy_eq):
    """Compute full performance metrics for strategy vs benchmark."""
    print("\n[5/5] Computing performance metrics...")

    r         = equity["daily_return"]
    total_ret = equity["portfolio"].iloc[-1] / INITIAL_CAPITAL - 1
    n_years   = len(equity) / TRADING_DAYS
    cagr      = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol   = r.std() * np.sqrt(TRADING_DAYS)
    sharpe    = (r.mean() * TRADING_DAYS - RISK_FREE_RATE) / ann_vol

    # Max drawdown
    rolling_max = equity["portfolio"].cummax()
    drawdown    = (equity["portfolio"] - rolling_max) / rolling_max
    max_dd      = drawdown.min()

    # Win rate (days with positive return when invested)
    invested_returns = equity[equity["invested"]]["daily_return"]
    win_rate = (invested_returns > 0).mean()

    # SPY metrics
    spy_r       = spy_eq["spy"].pct_change().fillna(0)
    spy_total   = spy_eq["spy"].iloc[-1] / INITIAL_CAPITAL - 1
    spy_cagr    = (1 + spy_total) ** (1 / n_years) - 1
    spy_vol     = spy_r.std() * np.sqrt(TRADING_DAYS)
    spy_sharpe  = (spy_r.mean() * TRADING_DAYS - RISK_FREE_RATE) / spy_vol
    spy_max_dd  = ((spy_eq["spy"] - spy_eq["spy"].cummax()) /
                    spy_eq["spy"].cummax()).min()

    metrics = {
        "Strategy": {
            "Total Return"  : f"{total_ret:.2%}",
            "CAGR"          : f"{cagr:.2%}",
            "Ann. Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio"  : f"{sharpe:.3f}",
            "Max Drawdown"  : f"{max_dd:.2%}",
            "Win Rate"      : f"{win_rate:.2%}",
        },
        "SPY (B&H)": {
            "Total Return"  : f"{spy_total:.2%}",
            "CAGR"          : f"{spy_cagr:.2%}",
            "Ann. Volatility": f"{spy_vol:.2%}",
            "Sharpe Ratio"  : f"{spy_sharpe:.3f}",
            "Max Drawdown"  : f"{spy_max_dd:.2%}",
            "Win Rate"      : "N/A",
        },
    }

    results = pd.DataFrame(metrics)
    print(f"\n{results.to_string()}")

    results.to_csv("notebooks/backtest_metrics.csv")
    print("\n    Saved: notebooks/backtest_metrics.csv")
    return results, sharpe, max_dd


# ── Plot equity curve ─────────────────────────────────────────────────────────

def plot_equity_curve(equity, spy_eq, metrics_df):
    """Interactive equity curve with drawdown panel."""
    print("\n    Plotting equity curve...")

    merged = pd.merge(equity, spy_eq, on="date", how="inner")
    merged["strategy_norm"] = merged["portfolio"] / INITIAL_CAPITAL
    merged["spy_norm"]      = merged["spy"] / INITIAL_CAPITAL

    # Drawdown series
    roll_max  = merged["portfolio"].cummax()
    drawdown  = (merged["portfolio"] - roll_max) / roll_max

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=(
            "Portfolio Value — Strategy vs SPY",
            "Drawdown",
            "Daily Invested (1 = in market)",
        ),
    )

    # Strategy line
    fig.add_trace(go.Scatter(
        x=merged["date"], y=merged["strategy_norm"],
        name="AlphaLens Strategy",
        line=dict(color="#1D9E75", width=2),
    ), row=1, col=1)

    # SPY line
    fig.add_trace(go.Scatter(
        x=merged["date"], y=merged["spy_norm"],
        name="SPY Buy & Hold",
        line=dict(color="#888780", width=1.5, dash="dash"),
    ), row=1, col=1)

    # $1 reference line
    fig.add_hline(y=1.0, line_dash="dot", line_color="#D3D1C7",
                  row=1, col=1)

    # Drawdown area
    fig.add_trace(go.Scatter(
        x=merged["date"], y=drawdown,
        name="Drawdown",
        fill="tozeroy",
        line=dict(color="#D85A30", width=1),
        fillcolor="rgba(216,90,48,0.2)",
    ), row=2, col=1)

    # Invested indicator
    fig.add_trace(go.Bar(
        x=merged["date"],
        y=equity["invested"].astype(int),
        name="In market",
        marker_color="rgba(29,158,117,0.4)",
    ), row=3, col=1)

    fig.update_layout(
        template  = "plotly_white",
        height    = 700,
        title     = "AlphaLens Backtest — Full Strategy vs SPY",
        showlegend= True,
    )
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".1%", row=2, col=1)

    fig.write_html("notebooks/equity_curve.html")
    print("    Saved: notebooks/equity_curve.html")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  AlphaLens — Day 5b: Backtester")
    print("=" * 55)

    df, weights_df, model = load_backtest_data()
    signals_df            = generate_signals(df, model, weights_df)
    equity                = run_backtest(signals_df, weights_df)
    spy_eq                = get_spy_benchmark(signals_df)
    metrics_df, sharpe, max_dd = compute_metrics(equity, spy_eq)
    plot_equity_curve(equity, spy_eq, metrics_df)

    equity.to_csv("notebooks/equity_curve.csv", index=False)

    print("\n" + "=" * 55)
    print(f"  Strategy Sharpe : {sharpe:.3f}")
    print(f"  Max Drawdown    : {max_dd:.2%}")
    print("=" * 55)
    print("\nDay 5 complete — ready for Day 6 Streamlit dashboard!")
