import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "AlphaLens",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Load data (cached) ────────────────────────────────────────────────────────

@st.cache_data
def load_features():
    BASE_DIR = os.path.dirname(__file__)
    path = os.path.join(BASE_DIR, "data", "final_features.parquet")

    if not os.path.exists(path):
        st.error("❌ final_features.parquet not found. Using fallback sample data.")
        path = os.path.join(BASE_DIR, "data", "sample_features.parquet")

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_equity():
    return pd.read_csv("notebooks/equity_curve.csv",
                       parse_dates=["date"])

@st.cache_data
def load_backtest_metrics():
    return pd.read_csv("notebooks/backtest_metrics.csv", index_col=0)

@st.cache_data
def load_model_comparison():
    return pd.read_csv("notebooks/model_comparison.csv", index_col=0)

@st.cache_data
def load_portfolio_weights():
    return pd.read_csv("data/portfolio_weights.csv")

@st.cache_data
def load_options_comparison():
    return pd.read_csv("notebooks/options_comparison.csv")

@st.cache_data
def load_regime_performance():
    return pd.read_csv("notebooks/regime_performance.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/xgboost_tuned.pkl")

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("AlphaLens")
st.sidebar.markdown("*Quantitative Investment Research Platform*")
st.sidebar.divider()

df      = load_features()
TICKERS = sorted(df["ticker"].unique().tolist())
ticker  = st.sidebar.selectbox("Select Ticker", TICKERS, index=0)
st.sidebar.divider()
st.sidebar.markdown("**Model**")
st.sidebar.markdown("XGBoost (Optuna tuned)")
st.sidebar.markdown("ROC-AUC: **0.5916**")
st.sidebar.markdown("Accuracy @ 0.60: **62.66%**")
st.sidebar.divider()
st.sidebar.markdown(
    "Built by [Adi0015](https://github.com/Adi0015) · April 2026"
)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Signals",
    "🤖 ML Model",
    "💼 Portfolio",
    "🔁 Backtest",
    "⚙️ Options",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.header(f"Market Signals — {ticker}")

    ticker_df = df[df["ticker"] == ticker].sort_values("date")

    # ── Price + regime chart
    regime_colors = {"bull": "#1D9E75", "bear": "#D85A30", "sideways": "#888780"}
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=("Price + Regime", "RSI", "Volume Ratio"),
    )

    # Price
    fig.add_trace(go.Scatter(
        x=ticker_df["date"], y=ticker_df["close"],
        name="Price", line=dict(color="#1D9E75", width=1.5)
    ), row=1, col=1)

    # EMA lines
    fig.add_trace(go.Scatter(
        x=ticker_df["date"], y=ticker_df["ema_50"],
        name="EMA 50", line=dict(color="#534AB7", width=1, dash="dot")
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ticker_df["date"], y=ticker_df["ema_200"],
        name="EMA 200", line=dict(color="#D85A30", width=1, dash="dash")
    ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=ticker_df["date"], y=ticker_df["rsi"],
        name="RSI", line=dict(color="#534AB7", width=1)
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#D85A30",
                  annotation_text="Overbought", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#1D9E75",
                  annotation_text="Oversold", row=2, col=1)

    # Volume ratio
    fig.add_trace(go.Bar(
        x=ticker_df["date"],
        y=ticker_df["vol_ratio"],
        name="Vol ratio",
        marker_color=ticker_df["vol_ratio"].apply(
            lambda x: "#D85A30" if x > 2 else "#B4B2A9"
        ),
    ), row=3, col=1)
    fig.add_hline(y=2, line_dash="dash", line_color="#D85A30",
                  row=3, col=1)

    fig.update_layout(
        template="plotly_white", height=600, showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Regime + sentiment metrics
    col1, col2, col3, col4 = st.columns(4)
    latest = ticker_df.dropna().iloc[-1]

    with col1:
        regime = latest.get("regime", "unknown")
        color  = {"bull": "🟢", "bear": "🔴", "sideways": "🟡"}.get(regime, "⚪")
        st.metric("Market Regime", f"{color} {regime.upper()}")
    with col2:
        st.metric("RSI", f"{latest['rsi']:.1f}")
    with col3:
        st.metric("Sentiment", f"{latest['sentiment_mean']:.3f}")
    with col4:
        st.metric("Volume Ratio", f"{latest['vol_ratio']:.2f}x")

    # ── Regime performance table
    st.subheader("Regime Performance (SPY)")
    regime_perf = load_regime_performance()
    st.dataframe(regime_perf.style.format({
        "ann_return_%": "{:.2f}%",
        "ann_vol_%"   : "{:.2f}%",
        "sharpe"      : "{:.3f}",
    }), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ML MODEL
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.header("ML Return Predictor")

    # ── Model comparison
    st.subheader("Model Comparison")
    model_comp = load_model_comparison()
    st.dataframe(model_comp.style.highlight_max(
        subset=["ROC-AUC"], color="#E1F5EE"
    ), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("SHAP Feature Importance")
        st.image("notebooks/shap_importance.png", use_column_width=True)

    with col2:
        st.subheader("SHAP Beeswarm — Impact Direction")
        st.image("notebooks/shap_beeswarm.png", use_column_width=True)

    # ── Live prediction for selected ticker
    st.subheader(f"Live Prediction — {ticker}")
    model       = load_model()
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
    available   = [c for c in FEATURE_COLS if c in df.columns]
    latest_row  = (df[df["ticker"] == ticker]
                     .dropna(subset=available)
                     .sort_values("date")
                     .iloc[[-1]])

    if len(latest_row) > 0:
        proba  = model.predict_proba(latest_row[available])[0, 1]
        signal = "🟢 BULLISH" if proba >= 0.60 else (
                 "🔴 BEARISH" if proba <= 0.40 else "🟡 NEUTRAL")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bull Probability", f"{proba:.2%}")
        with col2:
            st.metric("Signal", signal)
        with col3:
            st.metric("Confidence",
                      "High ✅" if proba >= 0.60 or proba <= 0.40
                      else "Low ⚠️")

        # Confidence gauge
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = proba * 100,
            title = {"text": f"{ticker} Bull Probability (%)"},
            gauge = {
                "axis" : {"range": [0, 100]},
                "bar"  : {"color": "#1D9E75" if proba >= 0.5 else "#D85A30"},
                "steps": [
                    {"range": [0,  40], "color": "#FAECE7"},
                    {"range": [40, 60], "color": "#F1EFE8"},
                    {"range": [60, 100],"color": "#E1F5EE"},
                ],
                "threshold": {
                    "line" : {"color": "#534AB7", "width": 3},
                    "thickness": 0.75,
                    "value": 60,
                },
            },
        ))
        fig_gauge.update_layout(height=300, template="plotly_white")
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Confidence threshold analysis
    st.subheader("Accuracy by Confidence Threshold")
    conf_df = pd.read_csv("notebooks/confidence_analysis.csv")
    fig_conf = px.line(
        conf_df, x="threshold", y="accuracy",
        markers=True,
        labels={"threshold": "Confidence Threshold",
                "accuracy" : "Accuracy"},
        template="plotly_white",
    )
    fig_conf.add_hline(y=0.5, line_dash="dash",
                       line_color="#888780",
                       annotation_text="Random baseline")
    fig_conf.update_layout(height=350)
    st.plotly_chart(fig_conf, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.header("Portfolio Optimizer")

    weights_df = load_portfolio_weights()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Portfolio Weights")
        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(
            name="Max Sharpe (MVO)",
            x=weights_df["ticker"],
            y=weights_df["mvo_weight"],
            marker_color="#1D9E75",
        ))
        fig_w.add_trace(go.Bar(
            name="Black-Litterman",
            x=weights_df["ticker"],
            y=weights_df["bl_weight"],
            marker_color="#534AB7",
        ))
        fig_w.update_layout(
            barmode="group",
            template="plotly_white",
            height=400,
            yaxis=dict(tickformat=".0%"),
        )
        st.plotly_chart(fig_w, use_container_width=True)

    with col2:
        st.subheader("ML Bull Probability by Ticker")
        fig_p = px.bar(
            weights_df.sort_values("bull_proba", ascending=True),
            x="bull_proba", y="ticker",
            orientation="h",
            color="bull_proba",
            color_continuous_scale="RdYlGn",
            range_color=[0.4, 0.8],
            labels={"bull_proba": "Bull Probability",
                    "ticker"    : "Ticker"},
            template="plotly_white",
        )
        fig_p.add_vline(x=0.60, line_dash="dash",
                        line_color="#534AB7",
                        annotation_text="0.60 threshold")
        fig_p.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_p, use_container_width=True)

    # Weights table
    st.subheader("Weights Detail")
    st.dataframe(
        weights_df.style.format({
            "mvo_weight" : "{:.2%}",
            "bl_weight"  : "{:.2%}",
            "mv_weight"  : "{:.2%}",
            "bull_proba" : "{:.2%}",
        }),
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.header("Strategy Backtest")

    equity  = load_equity()
    metrics = load_backtest_metrics()

    # Metrics scorecard
    st.subheader("Performance vs SPY")
    col1, col2, col3, col4, col5 = st.columns(5)
    cols    = [col1, col2, col3, col4, col5]
    metrics_list = ["Total Return", "CAGR", "Sharpe Ratio",
                    "Max Drawdown", "Ann. Volatility"]

    for i, metric in enumerate(metrics_list):
        with cols[i]:
            strat_val = metrics.loc[metric, "Strategy"]
            spy_val   = metrics.loc[metric, "SPY (B&H)"]
            st.metric(
                label=metric,
                value=strat_val,
                delta=f"SPY: {spy_val}",
            )

    st.divider()

    # Equity curve
    equity["strategy_norm"] = equity["portfolio"] / equity["portfolio"].iloc[0]
    roll_max  = equity["portfolio"].cummax()
    drawdown  = (equity["portfolio"] - roll_max) / roll_max

    spy_df = (df[df["ticker"] == "SPY"]
                .sort_values("date")[["date","close"]]
                .drop_duplicates("date")
                .reset_index(drop=True))
    spy_df["spy_norm"] = spy_df["close"] / spy_df["close"].iloc[0]
    merged_eq = pd.merge(
        equity[["date","strategy_norm"]],
        spy_df[["date","spy_norm"]],
        on="date", how="inner"
    )

    fig_eq = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=("Normalized Portfolio Value", "Drawdown"),
    )
    fig_eq.add_trace(go.Scatter(
        x=merged_eq["date"], y=merged_eq["strategy_norm"],
        name="AlphaLens", line=dict(color="#1D9E75", width=2)
    ), row=1, col=1)
    fig_eq.add_trace(go.Scatter(
        x=merged_eq["date"], y=merged_eq["spy_norm"],
        name="SPY", line=dict(color="#888780", width=1.5, dash="dash")
    ), row=1, col=1)
    fig_eq.add_hline(y=1.0, line_dash="dot",
                     line_color="#D3D1C7", row=1, col=1)
    fig_eq.add_trace(go.Scatter(
        x=equity["date"], y=drawdown,
        fill="tozeroy",
        name="Drawdown",
        line=dict(color="#D85A30", width=1),
        fillcolor="rgba(216,90,48,0.2)",
    ), row=2, col=1)
    fig_eq.update_layout(
        template="plotly_white", height=550,
        title="AlphaLens Strategy vs SPY Buy & Hold"
    )
    fig_eq.update_yaxes(tickformat=".0%", row=1, col=1)
    fig_eq.update_yaxes(tickformat=".1%", row=2, col=1)
    st.plotly_chart(fig_eq, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — OPTIONS
# ─────────────────────────────────────────────────────────────────────────────

with tab5:
    st.header("Options Pricer")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Inputs")
        S        = st.number_input("Spot Price (S)", value=200.0, step=1.0)
        K        = st.number_input("Strike Price (K)", value=200.0, step=1.0)
        T        = st.slider("Time to Expiry (years)", 0.01, 2.0, 0.25, 0.01)
        r        = st.slider("Risk-Free Rate", 0.01, 0.10, 0.05, 0.005)
        sigma    = st.slider("Implied Volatility", 0.05, 1.0, 0.28, 0.01)
        opt_type = st.radio("Option Type", ["call", "put"])

    with col2:
        st.subheader("Pricing Results")

        from scipy.stats import norm

        def bs_price(S, K, T, r, sigma, opt_type):
            if T <= 0 or sigma <= 0:
                return 0.0
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            if opt_type == "call":
                return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

        def greeks(S, K, T, r, sigma, opt_type):
            if T <= 0 or sigma <= 0:
                return 0, 0, 0, 0
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            delta = norm.cdf(d1) if opt_type=="call" else norm.cdf(d1)-1
            gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
            theta = (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) -
                     r*K*np.exp(-r*T)*norm.cdf(d2 if opt_type=="call" else -d2)) / 365
            vega  = S*norm.pdf(d1)*np.sqrt(T)/100
            return delta, gamma, theta, vega

        price            = bs_price(S, K, T, r, sigma, opt_type)
        delta, gamma, theta, vega = greeks(S, K, T, r, sigma, opt_type)

        c1, c2, c3 = st.columns(3)
        c1.metric("Option Price", f"${price:.4f}")
        c2.metric("Delta (Δ)", f"{delta:.4f}")
        c3.metric("Gamma (Γ)", f"{gamma:.6f}")
        c1.metric("Theta (Θ/day)", f"${theta:.4f}")
        c2.metric("Vega (ν/1%)", f"${vega:.4f}")
        c3.metric("Moneyness",
                  "ITM" if (opt_type=="call" and S>K) or
                            (opt_type=="put"  and S<K)
                  else "OTM" if (opt_type=="call" and S<K) or
                                 (opt_type=="put"  and S>K)
                  else "ATM")

        # Payoff diagram
        strikes_range = np.linspace(S * 0.7, S * 1.3, 200)
        if opt_type == "call":
            payoff = np.maximum(strikes_range - K, 0) - price
        else:
            payoff = np.maximum(K - strikes_range, 0) - price

        fig_pay = go.Figure()
        fig_pay.add_trace(go.Scatter(
            x=strikes_range, y=payoff,
            fill="tozeroy",
            line=dict(color="#1D9E75" if opt_type=="call" else "#D85A30",
                      width=2),
            fillcolor="rgba(29,158,117,0.15)" if opt_type=="call"
                      else "rgba(216,90,48,0.15)",
            name="P&L at expiry",
        ))
        fig_pay.add_hline(y=0, line_dash="dash", line_color="#888780")
        fig_pay.add_vline(x=K, line_dash="dot", line_color="#534AB7",
                          annotation_text="Strike")
        fig_pay.add_vline(x=S, line_dash="dot", line_color="#888780",
                          annotation_text="Spot")
        fig_pay.update_layout(
            title   = f"{opt_type.upper()} Payoff Diagram at Expiry",
            xaxis_title = "Stock Price at Expiry",
            yaxis_title = "P&L ($)",
            template    = "plotly_white",
            height      = 350,
        )
        st.plotly_chart(fig_pay, use_container_width=True)

    # Market comparison table
    st.subheader("BS vs Real Market Prices (AAPL)")
    options_df = load_options_comparison()
    if len(options_df) > 0:
        st.dataframe(
            options_df[["type","strike","market","bs_price",
                         "mc_price","bs_error","delta","iv"]]
                      .head(10)
                      .style.format({
                          "market"  : "${:.2f}",
                          "bs_price": "${:.2f}",
                          "mc_price": "${:.2f}",
                          "bs_error": "${:.4f}",
                          "delta"   : "{:.4f}",
                          "iv"      : "{:.2%}",
                      }),
            use_container_width=True,
        )
