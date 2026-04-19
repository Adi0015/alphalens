# AlphaLens — Quantitative Investment Research Platform

> Built to answer one question: **can a machine learning model find a consistent edge in equity markets?**

**Live Demo:** `[coming Day 7]` &nbsp;|&nbsp; **GitHub:** [Adi0015/alphalens](https://github.com/Adi0015/alphalens)

**Stack:** Python · XGBoost · LightGBM · Optuna · SHAP · FinBERT · HMM · Streamlit · Plotly

---

## The Result

**Yes — with the right conditions.**

A raw XGBoost model predicts 5-day return direction at **57.6% accuracy** across all signals.
When we only act on high-confidence predictions (≥ 0.60 probability), accuracy rises to **62.7% on 37.7% of trades.**
The Hidden Markov Model identifies bull regimes with a Sharpe of **1.54** vs **0.58** in bear regimes — a 2.7× difference in risk-adjusted returns.

In quantitative finance, a consistent 2–3% edge above random is what systematic funds are built on.
This project finds that edge, validates it, and shows exactly *why* it exists using SHAP explainability.

---

## What This Project Actually Does

AlphaLens is a full research pipeline — not just a model. It answers the question above by doing what a junior quant analyst would do at a fund:

1. **Collect data** — 5 years of prices + macroeconomic indicators for 10 large-cap tickers
2. **Engineer signals** — 46 features: technical indicators, macro regime, NLP sentiment, interaction terms
3. **Find the edge** — train and compare 4 models, tune the best two with Optuna, measure edge by confidence tier
4. **Validate it** — backtest the regime-filtered strategy vs SPY, compute Sharpe and max drawdown
5. **Price derivatives** — Black-Scholes + Monte Carlo options pricer validated against real market prices
6. **Make it usable** — deploy as an interactive Streamlit dashboard anyone can run

---

## Why the Metrics Are Actually Strong

> Predicting stock direction is one of the hardest problems in applied ML. Markets are adversarial — every publicly known signal gets arbitraged away. Academic literature considers **anything above 55% ROC-AUC** on out-of-sample financial data to be statistically meaningful.

Here is what AlphaLens achieves on **held-out test data** (no training data touched):

| What we measure | Result |
|---|---|
| ROC-AUC (all signals) | 0.5916 |
| Accuracy (all signals) | 57.61% |
| Accuracy at 0.60 confidence | **62.66%** |
| Accuracy at 0.65 confidence | **62.83%** |
| Trades available at 0.60 threshold | 37.7% of all opportunities |
| Bull regime Sharpe (HMM) | **1.544** |
| Bear regime Sharpe (HMM) | 0.579 |

The confidence threshold analysis is the key insight: **the model knows when it doesn't know.** A real trading desk would only act on the high-confidence signals — which is exactly what the backtester does.

---

## Architecture

![AlphaLens Architecture](notebooks/architecture.svg)

```
Raw Data (yfinance + FRED API)
          │
          ▼
  data_ingestion.py       ← 5 years, 10 tickers, 5 macro series
          │
          ▼
  feature_engineering.py  ← 46 features + interaction terms
          │
          ▼
  sentiment_engine.py     ← FinBERT on financial headlines
          │
          ▼
  ml_model.py             ← 4 models, Optuna tuning, SHAP
          │
          ├── quant/regime_detection.py    ← HMM bull/bear/sideways
          ├── quant/options_pricer.py      ← Black-Scholes + Monte Carlo
          ├── quant/portfolio_optimizer.py ← Mean-Variance + Black-Litterman
          └── quant/backtester.py          ← Strategy vs SPY benchmark
                    │
                    ▼
                app.py               ← Streamlit dashboard (5 tabs)
```

---

## The Signal Stack — What Drives Predictions

### Layer 1 — Technical signals (what the price is doing)
EMA 12/26/50/200 crossovers, RSI + RSI divergence, MACD histogram, Bollinger %B, ATR, OBV, realized volatility, z-score mean reversion, Williams %R, overnight gap, distance from 52-week high/low, volume spike detection.

### Layer 2 — Macro regime (what the economy is doing)
Fed Funds Rate, CPI, 10Y/2Y yield spread (recession signal), unemployment. FRED data forward-filled daily via `merge_asof` with `direction="backward"` — correct time-series join that avoids look-ahead bias.

### Layer 3 — Sentiment (what the market is saying)
Financial news headlines scored by `ProsusAI/finbert` — a BERT model fine-tuned on financial text. Generic sentiment models score "Fed raised rates 75bps" as neutral. FinBERT correctly scores it as negative for equities.

### Layer 4 — Interaction terms (signal combinations)
- RSI × volume spike — oversold stock + unusual volume = high-probability reversal
- MACD × sentiment — technical momentum confirmed by news flow
- Z-score × RSI — double mean-reversion confirmation
- Yield spread × sentiment — macro environment amplifying or dampening news

---

## Model Development

Four models trained on identical time-series splits (cutoff: 2025-06-24, no shuffling):

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| **XGBoost (Optuna tuned)** | **0.5761** | **0.6773** | **0.5916** |
| Random Forest | 0.5393 | 0.6964 | 0.5725 |
| LightGBM (Optuna tuned) | 0.5413 | 0.7024 | 0.5723 |
| Logistic Regression | 0.5463 | 0.6415 | 0.5443 |

**Confidence threshold analysis:**

| Threshold | Trades | Coverage | Accuracy |
|---|---|---|---|
| 0.50 | 2010 | 100.0% | 57.61% |
| 0.55 | 1305 | 64.9% | 59.62% |
| 0.60 | 758 | 37.7% | 62.66% |
| 0.65 | 339 | 16.9% | 62.83% |

**Why XGBoost won:** level-wise tree growth with aggressive regularization (`reg_alpha=0.41`, `reg_lambda=0.94`) outperformed LightGBM's leaf-wise strategy on this dataset. Optuna searched 50 trials across 8 hyperparameters per model.

---

## Modules

### Module 1 — Data Pipeline (`data_ingestion.py`)

- **10 tickers:** AAPL MSFT GOOGL AMZN JPM GS SPY QQQ TSLA NVDA
- **5 macro series:** CPI, Fed Funds Rate, 10Y yield, 2Y yield, unemployment
- **Key decision:** `pd.merge_asof(direction="backward")` — joins monthly FRED data onto daily prices without look-ahead. Extended FRED fetch 6 months back so every market date has a prior macro observation.
- **Output:** 12,550 rows × 16 columns, zero unexpected nulls

### Module 2 — Feature Engineering (`feature_engineering.py`)

- 46 ML-ready features across trend, momentum, volatility, volume, macro, sentiment, and interaction categories
- Target: `1` if 5-day forward return > 0, strictly shifted to avoid leakage
- Final matrix: 10,040 rows × 58 columns after indicator warmup period

### Module 3 — Sentiment Engine (`sentiment_engine.py`)

- NewsAPI → 100 headlines per ticker → FinBERT batch inference
- Score = `P(positive) − P(negative)` ∈ [−1, +1]
- Aggregated to `sentiment_mean`, `sentiment_std`, `headline_count` per day per ticker
- Forward-filled on non-publication days

### Module 4 — ML Return Predictor (`ml_model.py`)

- 4-model comparison with strict time-series split
- Optuna hyperparameter tuning (50 trials × 2 models)
- Confidence threshold analysis — identifies when to trust the model
- SHAP beeswarm + importance plots
- Dynamic model saving: winner auto-named `xgboost_tuned.pkl`

### Module 5 — Quant Engine

| Script | What it does | Key result |
|---|---|---|
| `quant/regime_detection.py` | HMM 3 states — only trade in bull regime | Bull Sharpe 1.544 vs Bear 0.579 |
| `quant/options_pricer.py` | Black-Scholes + 10,000-path Monte Carlo | Median market error $0.10 on AAPL |
| `quant/portfolio_optimizer.py` | MVO + Black-Litterman on ML-filtered universe | Efficient frontier across selected tickers |
| `quant/backtester.py` | Full strategy vs SPY with transaction costs | Regime + confidence filtered signals |

**HMM Regime transition probabilities:**

| From → To | Bear | Sideways | Bull |
|---|---|---|---|
| Bear | 95.4% | 1.5% | 3.1% |
| Sideways | 1.5% | 98.5% | 0.0% |
| Bull | 2.7% | 0.0% | 97.3% |

Regimes are highly persistent — bull markets stay bull 97.3% of the time day-to-day.

### Module 6 — Streamlit Dashboard (`app.py`)

| Tab | What you see |
|---|---|
| Signals | Price chart + EMA overlay + RSI + volume ratio + regime metrics |
| ML | Model comparison + SHAP plots + live prediction gauge per ticker |
| Portfolio | MVO vs Black-Litterman weights + bull probability per ticker |
| Backtest | Equity curve vs SPY + drawdown + performance scorecard |
| Options | Interactive BS pricer + payoff diagram + Greeks + market comparison |

---

## EDA (`notebooks/eda.py`)

Five interactive Plotly charts (open as `.html` in any browser):

- **Price history** — 5-year chart, all 10 tickers
- **Yield curve** — 2Y vs 10Y + spread (inverted = recession signal)
- **Return distributions** — violin plots per ticker
- **Macro dashboard** — CPI, Fed rate, unemployment, 10Y over time
- **Correlation heatmap** — feature multicollinearity check

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data | `yfinance`, `fredapi`, `pandas`, `pyarrow` |
| Features | `ta`, `numpy` |
| NLP | `transformers` (FinBERT), `torch`, `newsapi-python` |
| ML | `xgboost`, `lightgbm`, `scikit-learn`, `shap`, `optuna` |
| Quant | `scipy`, `hmmlearn` |
| Viz | `plotly` |
| App | `streamlit` |
| Env | `python-dotenv`, `tqdm`, `joblib` |

---

## Setup

```bash
git clone git@github.com:Adi0015/alphalens.git
cd alphalens
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Add API keys
cp .env.example .env
# FRED_API_KEY=your_key
# NEWS_API_KEY=your_key

# Run the full pipeline
python data_ingestion.py
python feature_engineering.py
python sentiment_engine.py
python ml_model.py
python quant/regime_detection.py
python quant/options_pricer.py
python quant/portfolio_optimizer.py
python quant/backtester.py

# Launch dashboard
streamlit run app.py
```

API keys (both free):
- FRED: https://fred.stlouisfed.org/docs/api/api_key.html
- NewsAPI: https://newsapi.org

---

## Repository Structure

```
alphalens/
├── data_ingestion.py           # Market + macro pipeline
├── feature_engineering.py      # 46 features + interaction terms
├── sentiment_engine.py         # FinBERT sentiment scoring
├── ml_model.py                 # 4-model comparison + Optuna + SHAP
├── app.py                      # Streamlit dashboard (5 tabs)
├── quant/
│   ├── __init__.py
│   ├── regime_detection.py     # HMM — bull/bear/sideways labels
│   ├── options_pricer.py       # Black-Scholes + Monte Carlo + Greeks
│   ├── portfolio_optimizer.py  # MVO + Black-Litterman
│   └── backtester.py           # Strategy vs SPY benchmark
├── notebooks/
│   ├── eda.py                  # Plotly EDA charts
│   ├── architecture.svg        # Architecture diagram
│   ├── shap_importance.png     # SHAP feature importance
│   ├── shap_beeswarm.png       # SHAP feature direction
│   ├── model_comparison.csv    # 4-model results
│   ├── confidence_analysis.csv # Accuracy by threshold
│   ├── regime_performance.csv  # Sharpe by regime
│   ├── options_comparison.csv  # BS vs market prices
│   ├── backtest_metrics.csv    # Strategy vs SPY metrics
│   └── equity_curve.csv        # Daily portfolio values
├── models/
│   ├── README.md               # How to regenerate all models
│   ├── xgboost_tuned.pkl       # Winning model (auto-named)
│   ├── xgboost_tuned_params.pkl
│   ├── scaler.pkl
│   ├── hmm_model.pkl
│   ├── hmm_state_map.pkl
│   └── hmm_scaler.pkl
├── data/
│   └── portfolio_weights.csv   # MVO + BL weights
├── requirements.txt
├── .env.example
└── README.md
```

---

## Interview Talking Points

**On project focus** — *"The core question is simple: does a systematic ML signal have an edge in equity markets? Everything — the pipeline, features, sentiment, quant engine, dashboard — exists to answer that one question rigorously."*

**On the metrics** — *"57% raw accuracy sounds modest, but financial prediction benchmarks are different from ImageNet. Out-of-sample ROC-AUC of 0.59 on 5 years of equity data is statistically meaningful. More importantly, the confidence threshold analysis shows the model knows when it's confident — at 0.60 threshold we hit 62.7% accuracy, which is the signal a real trading desk would act on."*

**On engineering decisions** — *"Used `merge_asof` with `direction='backward'` for the macro join — the only correct way to join monthly data onto daily data without look-ahead bias. Strict time-series split with no shuffling. Optuna hyperparameter search across 8 parameters × 50 trials × 2 models."*

**On the HMM** — *"The Hidden Markov Model is particularly useful here because market regimes are persistent — the model shows bull markets stay bull 97.3% of the time day-to-day. That persistence is what makes regime filtering useful: once you identify the regime, it's likely to stay there long enough to act on."*

**On the options pricer** — *"Black-Scholes and Monte Carlo converge to within $0.003 on ATM options — that's the mathematical validation. Comparing against real AAPL market prices shows a median error of $0.10, which is within bid-ask spread for liquid contracts."*

---

## Resume Bullet Points

```
Built a full quant research pipeline ingesting 5 years of market + macroeconomic data;
engineered 46 features (technical indicators, macro regime, FinBERT sentiment, interaction
terms) with a strict time-series split to eliminate look-ahead bias.

Tuned XGBoost and LightGBM with Optuna (50 trials each); XGBoost achieved ROC-AUC 0.592 —
at 0.60 confidence threshold, accuracy rises to 62.7% on 37.7% of trades, demonstrating a
consistent directional edge.

Implemented a quant engine with Hidden Markov Model regime detection (bull Sharpe 1.54 vs
bear 0.58), Black-Scholes + Monte Carlo options pricer (median error $0.10 vs market), and a
regime-filtered backtester benchmarked against SPY.
```

---

*Built by Adi0015 · April 2026*
