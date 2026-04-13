import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm


# ── Black-Scholes ─────────────────────────────────────────────────────────────

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    S     = spot price
    K     = strike price
    T     = time to expiry in years
    r     = risk-free rate (annualized)
    sigma = implied volatility (annualized)
    Returns theoretical option price.
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return round(float(price), 4)


# ── Greeks ────────────────────────────────────────────────────────────────────

def compute_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Returns delta, gamma, theta (per day), vega (per 1% vol move).
    Same for both call and put except delta and theta.
    """
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1

    # Gamma (same for call and put)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta (per calendar day)
    base_theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        theta = (base_theta - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (base_theta + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    # Vega per 1% move in vol
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    return {
        "delta": round(float(delta), 4),
        "gamma": round(float(gamma), 6),
        "theta": round(float(theta), 4),
        "vega" : round(float(vega),  4),
    }


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def monte_carlo_price(S, K, T, r, sigma, option_type="call",
                      n_paths=10_000, seed=42):
    """
    Geometric Brownian Motion simulation.
    Returns (price, standard_error).
    """
    np.random.seed(seed)
    Z   = np.random.standard_normal(n_paths)
    ST  = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    price = np.exp(-r * T) * np.mean(payoffs)
    se    = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)

    return round(float(price), 4), round(float(se), 6)


# ── Monte Carlo path plot ──────────────────────────────────────────────────────

def plot_mc_paths(S, K, T, r, sigma, option_type="call",
                  n_paths=200, n_steps=100, seed=42):
    """Plot sample GBM price paths."""
    np.random.seed(seed)
    dt   = T / n_steps
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = S

    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_paths)
        paths[t] = paths[t-1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )

    time_axis = np.linspace(0, T * 365, n_steps + 1)
    fig = go.Figure()

    for i in range(min(n_paths, 100)):
        fig.add_trace(go.Scatter(
            x=time_axis, y=paths[:, i],
            mode="lines",
            line=dict(width=0.5, color="rgba(29,158,117,0.2)"),
            showlegend=False,
        ))

    # Strike line
    fig.add_hline(y=K, line_dash="dash", line_color="#D85A30",
                  annotation_text=f"Strike K={K}")
    # Current price line
    fig.add_hline(y=S, line_dash="dot", line_color="#534AB7",
                  annotation_text=f"Spot S={S}")

    fig.update_layout(
        title=f"Monte Carlo GBM Paths — {option_type.capitalize()} "
              f"(S={S}, K={K}, T={T}y, σ={sigma})",
        xaxis_title="Days to expiry",
        yaxis_title="Stock price",
        template="plotly_white",
        height=450,
    )
    fig.write_html("notebooks/mc_paths.html")
    print("    Saved: notebooks/mc_paths.html")
    return fig


# ── Fetch real options ─────────────────────────────────────────────────────────

def fetch_market_options(ticker="AAPL"):
    """Fetch nearest-expiry options chain from yfinance."""
    print(f"\n[2/4] Fetching real options chain for {ticker}...")
    try:
        tk      = yf.Ticker(ticker)
        expiry  = tk.options[0]
        chain   = tk.option_chain(expiry)
        S       = tk.fast_info["last_price"]

        calls = chain.calls[["strike","lastPrice","impliedVolatility",
                              "volume","openInterest"]].copy()
        calls["type"]   = "call"
        puts  = chain.puts[["strike","lastPrice","impliedVolatility",
                             "volume","openInterest"]].copy()
        puts["type"]    = "put"

        options_df          = pd.concat([calls, puts]).reset_index(drop=True)
        options_df["expiry"]= expiry
        print(f"    Spot price : ${S:.2f}")
        print(f"    Expiry     : {expiry}")
        print(f"    Contracts  : {len(options_df)}")
        return options_df, S, expiry
    except Exception as e:
        print(f"    WARN: Could not fetch options — {e}")
        return None, None, None


# ── Compare BS vs market ──────────────────────────────────────────────────────

def compare_vs_market(ticker="AAPL", r=0.05):
    """
    Price top-volume contracts with Black-Scholes + Monte Carlo.
    Compare against real market prices.
    """
    print(f"\n[3/4] Pricing {ticker} options — BS + MC vs market...")
    options_df, S, expiry = fetch_market_options(ticker)

    if options_df is None:
        print("    Skipping — no options data available")
        return None

    import pandas as pd
    T = (pd.to_datetime(expiry) - pd.Timestamp.today()).days / 365
    T = max(T, 1/365)     # minimum 1 day

    # Filter to liquid contracts only
    liquid = options_df[
        (options_df["impliedVolatility"] > 0.05) &
        (options_df["volume"] > 5) &
        (options_df["lastPrice"] > 0.01)
    ].nlargest(30, "volume")

    records = []
    for _, row in liquid.iterrows():
        K, sigma, mkt, otype = (row["strike"], row["impliedVolatility"],
                                row["lastPrice"], row["type"])

        bs_price        = black_scholes(S, K, T, r, sigma, otype)
        mc_price, mc_se = monte_carlo_price(S, K, T, r, sigma, otype)
        greeks          = compute_greeks(S, K, T, r, sigma, otype)

        records.append({
            "ticker"  : ticker,
            "type"    : otype,
            "strike"  : K,
            "market"  : round(mkt, 4),
            "bs_price": bs_price,
            "mc_price": mc_price,
            "mc_se"   : mc_se,
            "bs_error": round(abs(bs_price - mkt), 4),
            "iv"      : round(sigma, 4),
            **greeks,
        })

    result = pd.DataFrame(records)
    print(f"\n    Top 5 contracts by volume:")
    print(result[["type","strike","market","bs_price",
                  "mc_price","bs_error","delta"]
                ].head(5).to_string(index=False))
    print(f"\n    Mean BS error : ${result['bs_error'].mean():.4f}")
    print(f"    Median BS error: ${result['bs_error'].median():.4f}")

    # Scatter: BS vs market
    fig = px.scatter(
        result, x="market", y="bs_price", color="type",
        hover_data=["strike","iv","delta"],
        labels={"market": "Market price ($)", "bs_price": "BS price ($)"},
        title=f"{ticker} — Black-Scholes vs Market Price",
        template="plotly_white",
        color_discrete_map={"call": "#1D9E75", "put": "#D85A30"},
    )
    fig.add_shape(type="line",
                  x0=result["market"].min(), y0=result["market"].min(),
                  x1=result["market"].max(), y1=result["market"].max(),
                  line=dict(dash="dash", color="gray"))
    fig.write_html("notebooks/bs_vs_market.html")
    print("    Saved: notebooks/bs_vs_market.html")

    return result


# ── Greeks heatmap ────────────────────────────────────────────────────────────

def plot_greeks_heatmap(S=200, r=0.05, T=0.25, option_type="call"):
    """2D heatmap of delta and gamma across strike × implied vol grid."""
    print("\n[4/4] Generating Greeks heatmap...")

    strikes = np.linspace(S * 0.70, S * 1.30, 30)
    vols    = np.linspace(0.10, 0.60, 30)

    deltas = np.zeros((len(vols), len(strikes)))
    gammas = np.zeros((len(vols), len(strikes)))
    thetas = np.zeros((len(vols), len(strikes)))
    vegas  = np.zeros((len(vols), len(strikes)))

    for i, sigma in enumerate(vols):
        for j, K in enumerate(strikes):
            g = compute_greeks(S, K, T, r, sigma, option_type)
            deltas[i, j] = g["delta"]
            gammas[i, j] = g["gamma"]
            thetas[i, j] = g["theta"]
            vegas[i, j]  = g["vega"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Delta", "Gamma", "Theta (per day)", "Vega (per 1% vol)"),
    )

    heatmaps = [
        (deltas, "RdYlGn", 1, 1),
        (gammas, "Blues",  1, 2),
        (thetas, "RdBu_r", 2, 1),
        (vegas,  "Purples",2, 2),
    ]

    for data, cscale, row, col in heatmaps:
        fig.add_trace(go.Heatmap(
            z=data,
            x=np.round(strikes, 1),
            y=np.round(vols, 2),
            colorscale=cscale,
            showscale=True,
        ), row=row, col=col)

    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text="Strike ($)", row=row, col=col)
            fig.update_yaxes(title_text="IV", row=row, col=col)

    fig.update_layout(
        title   = f"Options Greeks Heatmap — {option_type.capitalize()} "
                  f"(S={S}, T={T}y, r={r})",
        template= "plotly_white",
        height  = 700,
    )
    fig.write_html("notebooks/greeks_heatmap.html")
    print("    Saved: notebooks/greeks_heatmap.html")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  AlphaLens — Day 4b: Options Pricer")
    print("=" * 55)

    # Example pricing
    S, K, T, r, sigma = 200.0, 200.0, 0.25, 0.05, 0.28

    print("\n[1/4] Pricing example — ATM option:")
    for otype in ["call", "put"]:
        bs       = black_scholes(S, K, T, r, sigma, otype)
        mc, se   = monte_carlo_price(S, K, T, r, sigma, otype)
        greeks   = compute_greeks(S, K, T, r, sigma, otype)
        print(f"\n    {otype.upper():4s} | S={S}  K={K}  T={T}y  σ={sigma}")
        print(f"    Black-Scholes : ${bs}")
        print(f"    Monte Carlo   : ${mc} ± {se}")
        print(f"    Greeks        : Δ={greeks['delta']}  "
              f"Γ={greeks['gamma']}  "
              f"Θ={greeks['theta']}  "
              f"ν={greeks['vega']}")

    # Monte Carlo path plot
    plot_mc_paths(S=S, K=K, T=T, r=r, sigma=sigma, option_type="call")

    # Real market comparison
    comparison = compare_vs_market(ticker="AAPL", r=0.05)
    if comparison is not None:
        comparison.to_csv("notebooks/options_comparison.csv", index=False)
        print("    Saved: notebooks/options_comparison.csv")

    # Greeks heatmap (all 4 Greeks)
    plot_greeks_heatmap(S=200, r=0.05, T=0.25, option_type="call")

    print("\n" + "=" * 55)
    print("  Day 4b complete!")
    print("  Outputs saved to notebooks/")
    print("=" * 55)
