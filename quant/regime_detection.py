import numpy as np
import pandas as pd
import joblib
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Config ────────────────────────────────────────────────────────────────────

N_STATES     = 3
N_ITER       = 1000
RANDOM_SEED  = 42


# ── Train HMM ─────────────────────────────────────────────────────────────────

def train_hmm(df, ticker="SPY"):
    print(f"\n[1/4] Training HMM on {ticker} (market proxy)...")

    spy = (df[df["ticker"] == ticker]
             .sort_values("date")
             .copy()
             .dropna(subset=["daily_return", "realized_vol_20"]))

    X = spy[["daily_return", "realized_vol_20"]].values

    # Normalize features so HMM doesn't fixate on scale differences
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_model  = None
    best_score  = -np.inf

    # Run multiple random restarts — pick best log-likelihood
    for seed in range(20):
        try:
            m = GaussianHMM(
                n_components    = N_STATES,
                covariance_type = "diag",   # more stable than "full" on small datasets
                n_iter          = 2000,
                random_state    = seed,
                tol             = 1e-4,
            )
            m.fit(X_scaled)
            score = m.score(X_scaled)
            if score > best_score:
                best_score = score
                best_model = m
        except Exception:
            continue

    model         = best_model
    hidden_states = model.predict(X_scaled)
    spy["hmm_state"] = hidden_states

    # Label by mean UNSCALED daily return
    state_means = {}
    for state in range(N_STATES):
        mask = spy["hmm_state"] == state
        state_means[state] = spy.loc[mask, "daily_return"].mean()

    sorted_states = sorted(state_means, key=state_means.get)
    state_map = {
        sorted_states[0]: "bear",
        sorted_states[1]: "sideways",
        sorted_states[2]: "bull",
    }
    spy["regime"] = spy["hmm_state"].map(state_map)

    print(f"    Regime distribution:")
    print(spy["regime"].value_counts().to_string())
    print(f"\n    Regime stats (daily return):")
    print(spy.groupby("regime")["daily_return"]
            .agg(["mean", "std", "count"])
            .round(5).to_string())

    return model, spy, state_map, scaler


# ── Apply to all tickers ───────────────────────────────────────────────────────

def apply_regimes(df, spy_regimes):
    print("\n[2/4] Broadcasting regime labels to all tickers...")
    regime_map    = spy_regimes.set_index("date")["regime"].to_dict()
    df["regime"]  = df["date"].map(regime_map)
    df["is_bull"] = (df["regime"] == "bull").astype(int)
    null_pct = df["regime"].isnull().mean() * 100
    print(f"    Regime null %  : {null_pct:.1f}%")
    print(f"    Distribution   :\n{df['regime'].value_counts().to_string()}")
    return df


# ── Transition matrix ──────────────────────────────────────────────────────────

def print_transition_matrix(model, state_map):
    """Print the HMM regime transition probabilities."""
    print("\n    Regime transition probabilities:")
    inv_map  = {v: k for k, v in state_map.items()}
    regimes  = ["bear", "sideways", "bull"]
    trans    = model.transmat_

    header = f"{'':>12}" + "".join(f"{r:>12}" for r in regimes)
    print(f"    {header}")
    for from_r in regimes:
        from_s = inv_map[from_r]
        row    = "".join(
            f"{trans[from_s, inv_map[to_r]]:>12.4f}"
            for to_r in regimes
        )
        print(f"    {from_r:>12}{row}")


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_regimes(spy_df):
    """Price chart with regime background shading + return bars."""
    print("\n[3/4] Generating regime chart...")

    colors = {
        "bull"    : "rgba(29,158,117,0.15)",
        "bear"    : "rgba(216,90,48,0.15)",
        "sideways": "rgba(136,135,128,0.10)",
    }

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("SPY Price + Market Regime", "Daily Return"),
        row_heights=[0.7, 0.3],
    )

    # Price line
    fig.add_trace(go.Scatter(
        x=spy_df["date"], y=spy_df["close"],
        name="SPY", line=dict(color="#1D9E75", width=1.5)
    ), row=1, col=1)

    # Regime shading
    prev_regime = None
    start_date  = None
    for _, row in spy_df.iterrows():
        if row["regime"] != prev_regime:
            if prev_regime is not None:
                fig.add_vrect(
                    x0=start_date, x1=row["date"],
                    fillcolor=colors.get(prev_regime, "rgba(0,0,0,0)"),
                    layer="below", line_width=0,
                )
            start_date  = row["date"]
            prev_regime = row["regime"]

    # Return bars
    fig.add_trace(go.Bar(
        x=spy_df["date"],
        y=spy_df["daily_return"],
        name="Daily return",
        marker_color=spy_df["daily_return"].apply(
            lambda x: "#1D9E75" if x >= 0 else "#D85A30"
        ),
    ), row=2, col=1)

    # Legend annotation
    for regime, color in colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=color.replace("0.15", "0.6")
                                         .replace("0.10", "0.6")),
            name=regime.capitalize(),
        ), row=1, col=1)

    fig.update_layout(
        template    = "plotly_white",
        height      = 600,
        title       = "Market Regime Detection — Hidden Markov Model (3 states)",
        showlegend  = True,
    )
    fig.write_html("notebooks/regime_chart.html")
    print("    Saved: notebooks/regime_chart.html")
    return fig


# ── Regime performance summary ─────────────────────────────────────────────────

def regime_performance(spy_df):
    """Show annualized return and Sharpe by regime."""
    print("\n    Regime performance summary:")
    rows = []
    for regime in ["bull", "sideways", "bear"]:
        subset = spy_df[spy_df["regime"] == regime]["daily_return"].dropna()
        ann_return = subset.mean() * 252
        ann_vol    = subset.std() * np.sqrt(252)
        sharpe     = ann_return / ann_vol if ann_vol > 0 else 0
        rows.append({
            "regime"        : regime,
            "days"          : len(subset),
            "ann_return_%"  : round(ann_return * 100, 2),
            "ann_vol_%"     : round(ann_vol * 100, 2),
            "sharpe"        : round(sharpe, 3),
        })
    perf = pd.DataFrame(rows)
    print(perf.to_string(index=False))
    perf.to_csv("notebooks/regime_performance.csv", index=False)
    print("    Saved: notebooks/regime_performance.csv")
    return perf


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  AlphaLens — Day 4a: HMM Regime Detection")
    print("=" * 55)

    df = pd.read_parquet("data/final_features.parquet")
    df["date"] = pd.to_datetime(df["date"])

    model, spy_df, state_map, scaler = train_hmm(df, ticker="SPY")
    print_transition_matrix(model, state_map)
    df = apply_regimes(df, spy_df)
    plot_regimes(spy_df)
    regime_performance(spy_df)

    df.to_parquet("data/final_features.parquet", index=False)
    joblib.dump(model,     "models/hmm_model.pkl")
    joblib.dump(state_map, "models/hmm_state_map.pkl")
    joblib.dump(scaler,    "models/hmm_scaler.pkl")

    print(f"\n[4/4] Saved:")
    print(f"    data/final_features.parquet")
    print(f"    models/hmm_model.pkl")
    print(f"    models/hmm_state_map.pkl")
    print(f"    models/hmm_scaler.pkl")
    print("\nDay 4a complete!")
