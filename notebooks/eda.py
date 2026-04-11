import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df    = pd.read_parquet("data/merged_data.parquet")
macro = pd.read_parquet("data/macro_data.parquet")

# ── 1. Price history ──────────────────────────────────────────────────────────

fig1 = px.line(
    df, x="date", y="close", color="ticker",
    title="5-Year Price History",
    labels={"close": "Price (USD)", "date": "Date"},
    template="plotly_white"
)
fig1.update_layout(legend=dict(orientation="h", y=-0.2))
fig1.write_html("notebooks/price_history.html")

# ── 2. Yield curve ────────────────────────────────────────────────────────────

fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     subplot_titles=("2Y vs 10Y Treasury Yield", "Yield Spread (10Y - 2Y)"))

fig2.add_trace(go.Scatter(x=macro["date"], y=macro["yield_10y"],
               name="10Y", line=dict(color="#1f77b4")), row=1, col=1)
fig2.add_trace(go.Scatter(x=macro["date"], y=macro["yield_2y"],
               name="2Y",  line=dict(color="#ff7f0e")), row=1, col=1)
fig2.add_trace(go.Scatter(x=macro["date"], y=macro["yield_spread"],
               name="Spread", fill="tozeroy",
               line=dict(color="#2ca02c")), row=2, col=1)
fig2.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
fig2.update_layout(title="Yield Curve Over Time", template="plotly_white", height=500)
fig2.write_html("notebooks/yield_curve.html")

# ── 3. Return distributions ───────────────────────────────────────────────────

fig3 = px.violin(
    df.dropna(subset=["daily_return"]),
    x="ticker", y="daily_return", color="ticker",
    box=True, points=False,
    title="Daily Return Distribution by Ticker",
    labels={"daily_return": "Daily Return", "ticker": "Ticker"},
    template="plotly_white"
)
fig3.update_layout(showlegend=False)
fig3.write_html("notebooks/return_distributions.html")

# ── 4. Macro dashboard ────────────────────────────────────────────────────────

fig4 = make_subplots(rows=2, cols=2,
                     subplot_titles=("CPI", "Fed Funds Rate",
                                     "Unemployment", "10Y Yield"))
pairs = [("cpi", 1, 1), ("fed_rate", 1, 2),
         ("unemployment", 2, 1), ("yield_10y", 2, 2)]
for col, row, c in pairs:
    fig4.add_trace(go.Scatter(x=macro["date"], y=macro[col],
                   name=col, showlegend=False), row=row, col=c)
fig4.update_layout(title="Macro Indicators", template="plotly_white", height=500)
fig4.write_html("notebooks/macro_dashboard.html")

# ── 5. Correlation heatmap ────────────────────────────────────────────────────

corr_cols = ["daily_return", "cpi", "fed_rate", "yield_10y",
             "yield_2y", "yield_spread", "unemployment"]
corr = df[corr_cols].corr().round(2)

fig5 = px.imshow(
    corr, text_auto=True, color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="Feature Correlation Heatmap",
    template="plotly_white"
)
fig5.write_html("notebooks/correlation_heatmap.html")

# ── Summary stats ─────────────────────────────────────────────────────────────

print("=" * 50)
print("Return summary by ticker:")
print(df.groupby("ticker")["daily_return"].describe().round(4))
print("\nAll charts saved to notebooks/ as interactive HTML files.")
