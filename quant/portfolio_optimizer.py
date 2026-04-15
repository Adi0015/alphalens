import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize


# ── Config ────────────────────────────────────────────────────────────────────

RISK_FREE_RATE     = 0.05        # annualized
N_PORTFOLIOS       = 5000        # Monte Carlo portfolios for frontier
MIN_CONFIDENCE     = 0.58       # only optimize over high-confidence tickers
TRADING_DAYS       = 252


# ── Load ML predictions ───────────────────────────────────────────────────────

def get_bullish_tickers(feature_path="data/final_features.parquet",
                        model_path="models/xgboost_tuned.pkl"):
    """
    Load the trained XGBoost model, run predictions on latest data,
    return tickers where model is bullish with high confidence.
    """
    print("\n[1/5] Loading ML model and identifying bullish tickers...")

    df    = pd.read_parquet(feature_path)
    model = joblib.load(model_path)

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

    available = [c for c in FEATURE_COLS if c in df.columns]

    # Use latest available date per ticker
    latest = (df.sort_values("date")
                .groupby("ticker")
                .last()
                .reset_index()
                .dropna(subset=available))

    X     = latest[available]
    proba = model.predict_proba(X)[:, 1]
    latest["bull_proba"] = proba
    latest["signal"]     = (proba >= MIN_CONFIDENCE).astype(int)

    bullish = latest[latest["signal"] == 1][["ticker","bull_proba"]].copy()
    bullish = bullish.sort_values("bull_proba", ascending=False)

    # Always include SPY as a defensive anchor
    if "SPY" not in bullish["ticker"].tolist():
        spy_row = latest[latest["ticker"] == "SPY"][["ticker","bull_proba"]]
        bullish = pd.concat([bullish, spy_row]).reset_index(drop=True)

    print(f"    Selected : {bullish['ticker'].tolist()}")
    return bullish["ticker"].tolist(), bullish



# ── Build return matrix ────────────────────────────────────────────────────────

def build_return_matrix(tickers, feature_path="data/final_features.parquet"):
    """Daily returns matrix for selected tickers."""
    print("\n[2/5] Building return matrix...")

    df = pd.read_parquet(feature_path)
    df["date"] = pd.to_datetime(df["date"])

    # Only use bull regime days for return estimation
    pivot = (df[df["ticker"].isin(tickers)]
               .pivot(index="date", columns="ticker", values="daily_return")
               .dropna())

    print(f"    Return matrix   : {pivot.shape}")
    print(f"    Date range      : {pivot.index.min()} → {pivot.index.max()}")
    return pivot


# ── MVO functions ─────────────────────────────────────────────────────────────

def portfolio_stats(weights, returns):
    """Annualized return, volatility, Sharpe for a weight vector."""
    port_return = np.dot(weights, returns.mean()) * TRADING_DAYS
    port_vol    = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov() * TRADING_DAYS, weights))
    )
    sharpe      = (port_return - RISK_FREE_RATE) / port_vol
    return port_return, port_vol, sharpe


def max_sharpe_weights(returns):
    """Find weights that maximize Sharpe ratio via scipy minimize."""
    n = returns.shape[1]

    def neg_sharpe(w):
        r, v, s = portfolio_stats(w, returns)
        return -s

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds      = tuple((0.0, 1.0) for _ in range(n))
    init        = np.array([1 / n] * n)

    result = minimize(neg_sharpe, init,
                      method="SLSQP",
                      bounds=bounds,
                      constraints=constraints,
                      options={"maxiter": 1000})
    return result.x


def min_variance_weights(returns):
    """Find weights that minimize portfolio variance."""
    n = returns.shape[1]

    def port_variance(w):
        return np.dot(w.T, np.dot(returns.cov() * TRADING_DAYS, w))

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds      = tuple((0.0, 1.0) for _ in range(n))
    init        = np.array([1 / n] * n)

    result = minimize(port_variance, init,
                      method="SLSQP",
                      bounds=bounds,
                      constraints=constraints)
    return result.x


# ── Efficient frontier ────────────────────────────────────────────────────────

def simulate_frontier(returns, n=N_PORTFOLIOS):
    """Monte Carlo simulation of random portfolios to map the frontier."""
    print(f"\n[3/5] Simulating {n} random portfolios...")
    n_assets = returns.shape[1]
    results  = []

    for _ in range(n):
        w       = np.random.dirichlet(np.ones(n_assets))
        r, v, s = portfolio_stats(w, returns)
        results.append({"return": r, "vol": v, "sharpe": s,
                         "weights": w.tolist()})

    frontier = pd.DataFrame(results)
    print(f"    Sharpe range: {frontier['sharpe'].min():.3f} → "
          f"{frontier['sharpe'].max():.3f}")
    return frontier


# ── Black-Litterman tilt ───────────────────────────────────────────────────────

def black_litterman_weights(returns, bullish_df, tickers):
    """
    Simplified Black-Litterman:
    Use ML bull_proba as 'views' to tilt weights away from pure MVO.
    Higher confidence = higher weight tilt.
    """
    print("\n[4/5] Applying Black-Litterman views from ML model...")

    # Start from equal weights
    n           = len(tickers)
    base        = np.array([1 / n] * n)

    # ML view: bull_proba relative to 0.5 baseline
    proba_map   = bullish_df.set_index("ticker")["bull_proba"]
    views       = np.array([proba_map.get(t, 0.5) - 0.5 for t in tickers])

    # Tilt toward higher-confidence tickers
    bl_weights  = base + 0.3 * views
    bl_weights  = np.clip(bl_weights, 0.02, 0.50)   # min 2%, max 50% per asset
    bl_weights /= bl_weights.sum()                   # renormalize

    print(f"    BL weights: "
          f"{dict(zip(tickers, bl_weights.round(4)))}")
    return bl_weights


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_frontier(frontier, max_sharpe_ret, max_sharpe_vol,
                  min_var_ret, min_var_vol, tickers):
    """Interactive efficient frontier with optimal portfolios marked."""
    print("\n    Plotting efficient frontier...")

    fig = go.Figure()

    # Scatter of random portfolios coloured by Sharpe
    fig.add_trace(go.Scatter(
        x=frontier["vol"],
        y=frontier["return"],
        mode="markers",
        marker=dict(
            color=frontier["sharpe"],
            colorscale="Viridis",
            size=3,
            opacity=0.5,
            colorbar=dict(title="Sharpe"),
        ),
        name="Random portfolios",
        hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<br>",
    ))

    # Max Sharpe point
    fig.add_trace(go.Scatter(
        x=[max_sharpe_vol], y=[max_sharpe_ret],
        mode="markers",
        marker=dict(color="#1D9E75", size=14, symbol="star"),
        name="Max Sharpe",
    ))

    # Min Variance point
    fig.add_trace(go.Scatter(
        x=[min_var_vol], y=[min_var_ret],
        mode="markers",
        marker=dict(color="#D85A30", size=14, symbol="diamond"),
        name="Min Variance",
    ))

    fig.update_layout(
        title   = f"Efficient Frontier — {', '.join(tickers)}",
        xaxis_title = "Annualized Volatility",
        yaxis_title = "Annualized Return",
        template    = "plotly_white",
        height      = 550,
        xaxis       = dict(tickformat=".0%"),
        yaxis       = dict(tickformat=".0%"),
    )
    fig.write_html("notebooks/efficient_frontier.html")
    print("    Saved: notebooks/efficient_frontier.html")
    return fig


def plot_weights(tickers, ms_weights, bl_weights):
    """Bar chart comparing Max Sharpe vs Black-Litterman weights."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Max Sharpe (MVO)",
        x=tickers,
        y=ms_weights,
        marker_color="#1D9E75",
    ))
    fig.add_trace(go.Bar(
        name="Black-Litterman",
        x=tickers,
        y=bl_weights,
        marker_color="#534AB7",
    ))

    fig.update_layout(
        barmode     = "group",
        title       = "Portfolio Weights — MVO vs Black-Litterman",
        xaxis_title = "Ticker",
        yaxis_title = "Weight",
        template    = "plotly_white",
        height      = 450,
        yaxis       = dict(tickformat=".0%"),
    )
    fig.write_html("notebooks/portfolio_weights.html")
    print("    Saved: notebooks/portfolio_weights.html")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  AlphaLens — Day 5a: Portfolio Optimizer")
    print("=" * 55)

    tickers, bullish_df = get_bullish_tickers()

    if len(tickers) < 2:
        print("    Not enough bullish tickers — using all tickers")
        df      = pd.read_parquet("data/final_features.parquet")
        tickers = df["ticker"].unique().tolist()
        bullish_df = pd.DataFrame({
            "ticker"    : tickers,
            "bull_proba": [0.5] * len(tickers)
        })

    returns = build_return_matrix(tickers)

    # Optimize
    ms_weights  = max_sharpe_weights(returns)
    mv_weights  = min_variance_weights(returns)
    bl_weights  = black_litterman_weights(returns, bullish_df, tickers)

    ms_ret, ms_vol, ms_sharpe = portfolio_stats(ms_weights, returns)
    mv_ret, mv_vol, mv_sharpe = portfolio_stats(mv_weights, returns)
    bl_ret, bl_vol, bl_sharpe = portfolio_stats(bl_weights, returns)

    print(f"\n    Portfolio results:")
    print(f"    {'Strategy':<20} {'Return':>10} {'Vol':>10} {'Sharpe':>10}")
    print(f"    {'-'*52}")
    print(f"    {'Max Sharpe (MVO)':<20} {ms_ret:>10.2%} "
          f"{ms_vol:>10.2%} {ms_sharpe:>10.3f}")
    print(f"    {'Min Variance':<20} {mv_ret:>10.2%} "
          f"{mv_vol:>10.2%} {mv_sharpe:>10.3f}")
    print(f"    {'Black-Litterman':<20} {bl_ret:>10.2%} "
          f"{bl_vol:>10.2%} {bl_sharpe:>10.3f}")

    print(f"\n    Max Sharpe weights:")
    for t, w in zip(tickers, ms_weights):
        print(f"      {t:<8} {w:.2%}")

    # Simulate frontier
    frontier = simulate_frontier(returns)

    # Plots
    plot_frontier(frontier, ms_ret, ms_vol, mv_ret, mv_vol, tickers)
    plot_weights(tickers, ms_weights, bl_weights)

    # Save weights for backtester
    weights_df = pd.DataFrame({
        "ticker"        : tickers,
        "mvo_weight"    : ms_weights,
        "bl_weight"     : bl_weights,
        "mv_weight"     : mv_weights,
        "bull_proba"    : [bullish_df.set_index("ticker")
                           ["bull_proba"].get(t, 0.5) for t in tickers],
    })
    weights_df.to_csv("data/portfolio_weights.csv", index=False)
    print(f"\n    Saved: data/portfolio_weights.csv")
    joblib.dump({"tickers": tickers,
                 "ms_weights": ms_weights,
                 "bl_weights": bl_weights},
                "models/portfolio.pkl")
    print(f"    Saved: models/portfolio.pkl")
    print("\nDay 5a complete — run backtester.py next!")
