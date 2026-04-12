# AlphaLens — End-to-End Quantitative Investment Platform

> A full-stack quant finance project combining data science, machine learning, and quantitative finance into one unified, deployed application.

**Live Demo:** `[coming Day 7]` &nbsp;|&nbsp; **Stack:** Python · Pandas · XGBoost · FinBERT · Plotly · Streamlit

---

## What is AlphaLens?

AlphaLens is a personal quantitative investment research platform built from scratch. It ingests 5 years of real market and macroeconomic data, engineers 46 features, scores financial news sentiment using a fine-tuned NLP model, trains a machine learning model to predict short-term returns, and ties everything together in an interactive Streamlit dashboard — all in one codebase.

The project deliberately covers three domains in one place:

- **Data Science** — real-world data ingestion, cleaning, and feature engineering
- **Machine Learning** — supervised classification with explainability (SHAP)
- **Quantitative Finance** — options pricing, portfolio optimization, backtesting

---

## Project Architecture

![AlphaLens Architecture](notebooks/architecture.svg)

```
Raw Data Sources
      │
      ├── yfinance (OHLCV, 10 tickers, 5 years)
      └── FRED API (CPI, Fed rate, yields, unemployment)
            │
            ▼
    data_ingestion.py
    (merge_asof time-series join)
            │
            ▼
    feature_engineering.py
    (RSI, MACD, Bollinger, volume ratios, macro features, interaction terms, target variable)
            │
            ▼
    sentiment_engine.py
    (NewsAPI headlines → FinBERT → daily sentiment score per ticker)
            │
            ▼
    ml_model.py
    (XGBoost + LightGBM tuned via Optuna → SHAP explainability)
            │
            ├── regime_detection.py (Hidden Markov Model)
            ├── options_pricer.py   (Black-Scholes + Monte Carlo)
            ├── portfolio_optimizer.py (Mean-Variance + Black-Litterman)
            └── backtester.py       (vectorized strategy vs SPY benchmark)
                        │
                        ▼
                    app.py
              (Streamlit Dashboard)
```

---

## Modules

### Module 1 — Data Pipeline (`data_ingestion.py`)

Pulls and merges two data sources into clean Parquet files:

**Market data via yfinance**
- 10 tickers: `AAPL MSFT GOOGL AMZN JPM GS SPY QQQ TSLA NVDA`
- 5 years of daily OHLCV data (open, high, low, close, volume)
- Derived columns: `daily_return`, `log_return`

**Macro data via FRED API**
- CPI (inflation), Federal Funds Rate, 10Y & 2Y Treasury Yields, Unemployment Rate
- Monthly FRED observations forward-filled to business days
- Derived: `yield_spread` (10Y − 2Y), `cpi_mom` (month-over-month CPI change)

**Key engineering decision:** used `pd.merge_asof` with `direction="backward"` to join monthly macro data onto daily market data — the correct approach for time-series data that avoids look-ahead bias. Extended FRED fetch window by 6 months before market start date to eliminate early-date nulls where no prior observation existed.

Output files:
```
data/
├── market_data.parquet    (~12,560 rows × 9 cols)
├── macro_data.parquet     (~1,300 rows × 8 cols)
└── merged_data.parquet    (~12,550 rows × 16 cols)
```

---

### Module 2 — Feature Engineering (`feature_engineering.py`)

Computes 46 features per ticker per day across 5 categories:

**Trend indicators**
- EMA 12, EMA 26, EMA 50, EMA 200
- MACD, MACD Signal, MACD Histogram
- Above EMA 50/200 flags, EMA 50/200 crossover signal

**Momentum indicators**
- RSI (14-day), RSI 10-day MA, RSI divergence
- Rate of Change (10-day)
- Williams %R

**Volatility indicators**
- Bollinger Bands (upper, lower, width, %B)
- Average True Range (ATR)
- Realized volatility (20-day annualized)
- Z-score mean reversion signal (20-day)

**Volume indicators**
- On-Balance Volume (OBV)
- Volume vs 20-day moving average ratio
- Volume spike flag (>2× average)

**Price-derived features**
- 5-day and 20-day rolling returns
- Distance from 52-week high and low
- Overnight gap (open vs prior close)
- Price vs SPY relative strength

**Interaction features**
- RSI × volume spike — oversold + high volume = reversal signal
- MACD × sentiment — technical + news alignment
- Volatility × momentum — momentum in high-vol environment
- Z-score × RSI — double mean-reversion confirmation
- Yield spread × sentiment — macro + news alignment

**Target variable**
- `target = 1` if price is higher 5 trading days from now, else `0`
- Strictly uses future data shifted by 5 days — no look-ahead leakage
- Final feature matrix: **10,040 rows × 58 columns** (after dropping NaNs from indicators)
- 46 features used for ML training

---

### Module 3 — Sentiment Engine (`sentiment_engine.py`)

Scores daily financial news sentiment per ticker using FinBERT:

1. Fetches up to 100 recent headlines per ticker via NewsAPI
2. Passes headlines through `ProsusAI/finbert` — a BERT model fine-tuned on financial text
3. Computes sentiment score = `P(positive) − P(negative)` in range `[−1, +1]`
4. Aggregates to daily level: `sentiment_mean`, `sentiment_std`, `headline_count`
5. Forward-fills into the feature matrix (weekend/missing days carry last score)

FinBERT understands financial language that generic sentiment models miss — e.g. "the Fed raised rates by 75bps" is correctly scored as negative for equities.

---

### Module 4 — ML Return Predictor (`ml_model.py`)

Trains and compares 4 models to predict 5-day return direction:

- **Winner:** XGBoost tuned via Optuna (50 trials)
- **Features:** 46 features — technical indicators, macro, sentiment, interaction terms
- **Split:** strict time-series split cutoff at 2025-06-24 — no shuffling (avoids leakage)
- **ROC-AUC:** 0.5916 | **Accuracy:** 57.61% (all trades)
- **At 0.60 confidence threshold:** 62.66% accuracy on 37.7% of trades
- **Explainability:** SHAP summary + beeswarm plots saved to `notebooks/`
- **Dynamic model saving:** winning model auto-named `xgboost_tuned.pkl`

**Model comparison results:**

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| XGBoost (tuned) | 0.5761 | 0.6773 | 0.5916 |
| Random Forest | 0.5393 | 0.6964 | 0.5725 |
| LightGBM (tuned) | 0.5413 | 0.7024 | 0.5723 |
| Logistic Regression | 0.5463 | 0.6415 | 0.5443 |

**Confidence threshold analysis:**

| Threshold | Trades | Coverage | Accuracy |
|---|---|---|---|
| 0.50 | 2010 | 100.0% | 57.61% |
| 0.55 | 1305 | 64.9% | 59.62% |
| 0.60 | 758 | 37.7% | 62.66% |
| 0.65 | 339 | 16.9% | 62.83% |

---

### Module 5 — Quant Engine

**Regime Detection** (`regime_detection.py`)
- Hidden Markov Model (3 states: bull / bear / sideways)
- Labels each trading day with a market regime
- Used to condition the ML strategy (only trade in bull regime)

**Options Pricer** (`options_pricer.py`)
- Black-Scholes analytical pricer for calls and puts
- Monte Carlo simulation (1,000 paths) as validation
- Computes Greeks: Delta, Gamma, Theta, Vega
- Compared against real market prices from yfinance options chain

**Portfolio Optimizer** (`portfolio_optimizer.py`)
- Mean-Variance Optimization (Markowitz efficient frontier)
- Filters the investable universe using ML model's top predicted tickers
- Outputs optimal portfolio weights that maximize Sharpe ratio

**Backtester** (`backtester.py`)
- Simulates the full ML + regime + optimization strategy historically
- Computes: Sharpe ratio, max drawdown, CAGR, win rate
- Benchmarks against buy-and-hold SPY

---

### Module 6 — Streamlit Dashboard (`app.py`)

Five interactive tabs, all built with Plotly:

| Tab | Content |
|-----|---------|
| Signals | Price chart with regime overlay + sentiment timeline |
| ML | Prediction table + SHAP waterfall chart |
| Portfolio | Efficient frontier + portfolio weights |
| Backtest | Equity curve vs SPY + performance metrics |
| Options | Greeks heatmap + Monte Carlo paths |

---

## EDA Highlights

Five exploratory charts are generated by `notebooks/eda.py` (all interactive Plotly HTML):

- **Price history** — 5-year normalized price chart for all 10 tickers
- **Yield curve** — 2Y vs 10Y treasury yields with spread (recession signal)
- **Return distributions** — violin plots showing daily return spread per ticker
- **Macro dashboard** — CPI, Fed rate, unemployment, 10Y yield over time
- **Correlation heatmap** — feature correlation matrix to spot multicollinearity

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data ingestion | `yfinance`, `fredapi`, `pandas`, `pyarrow` |
| Technical indicators | `ta` |
| NLP / Sentiment | `transformers` (FinBERT), `torch`, `newsapi-python` |
| Machine learning | `xgboost`, `lightgbm`, `scikit-learn`, `shap`, `optuna` |
| Regime detection | `hmmlearn` |
| Quant finance | `numpy`, `scipy`, `vectorbt` |
| Visualization | `plotly` |
| Dashboard | `streamlit` |
| Environment | `python-dotenv`, `tqdm` |

---

## Setup & Usage

```bash
# 1. Clone
git clone git@github.com:Adi0015/alphalens.git
cd alphalens

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add API keys
cp .env.example .env
# Edit .env and add:
# FRED_API_KEY=your_key_here
# NEWS_API_KEY=your_key_here

# 5. Run the pipeline
python data_ingestion.py       # Day 1 — fetch & merge data
python feature_engineering.py  # Day 2 — compute 46 features
python sentiment_engine.py     # Day 2 — FinBERT sentiment scores
python ml_model.py             # Day 3 — tune & compare 4 models

# 6. Launch dashboard
streamlit run app.py
```

---

## Project Timeline

Built in 7 days:

| Day | Focus | Output |
|-----|-------|--------|
| 1 | Data pipeline | 3 clean Parquet files, 12,550 rows of market + macro data |
| 2 | Feature engineering + sentiment | 58-column feature matrix with 46 ML features + FinBERT scores |
| 3 | ML model | XGBoost tuned via Optuna — ROC-AUC 0.5916, 62.66% accuracy at 0.60 threshold |
| 4 | Regime detection + options pricer | HMM labels + Black-Scholes Greeks |
| 5 | Portfolio optimizer + backtester | Efficient frontier + equity curve vs SPY |
| 6 | Streamlit dashboard | 5-tab interactive app |
| 7 | Deploy + polish | Live URL + complete README |

---

## Environment Variables

Create a `.env` file in the project root (never commit this):

```
FRED_API_KEY=your_fred_api_key
NEWS_API_KEY=your_newsapi_key
```

Get keys free at:
- FRED: https://fred.stlouisfed.org/docs/api/api_key.html
- NewsAPI: https://newsapi.org

---

## Repository Structure

```
alphalens/
├── data_ingestion.py          # Market + macro data pipeline
├── feature_engineering.py     # 46 technical, macro, sentiment, interaction features
├── sentiment_engine.py        # FinBERT sentiment scoring
├── ml_model.py                # 4-model comparison + Optuna tuning + SHAP
├── quant/
│   ├── regime_detection.py    # Hidden Markov Model
│   ├── options_pricer.py      # Black-Scholes + Monte Carlo
│   ├── portfolio_optimizer.py # Mean-Variance Optimization
│   └── backtester.py          # Strategy backtesting
├── notebooks/
│   ├── eda.py                 # Plotly exploratory analysis
│   ├── architecture.svg       # Project architecture diagram
│   ├── shap_importance.png    # SHAP feature importance plot
│   ├── shap_beeswarm.png      # SHAP beeswarm direction plot
│   ├── model_comparison.csv   # 4-model comparison results
│   └── confidence_analysis.csv # Accuracy by confidence threshold
├── models/
│   ├── xgboost_tuned.pkl      # Winning model (auto-named by algorithm)
│   ├── xgboost_tuned_params.pkl # Optuna best hyperparameters
│   └── scaler.pkl             # StandardScaler for Logistic Regression
├── app.py                     # Streamlit dashboard
├── requirements.txt
├── .env.example
└── README.md
```

---

## Interview Talking Points

**Data Science**
- Used `merge_asof` with `direction="backward"` — correct way to join monthly macro onto daily market data without look-ahead bias
- Extended FRED fetch window by 6 months before market start date to eliminate early-date nulls where no prior observation existed
- Only expected nulls are `daily_return` and `log_return` on each ticker's first row — mathematically unavoidable

**Machine Learning**
- Strict time-series train/test split — no shuffling to prevent look-ahead leakage
- Tuned both XGBoost and LightGBM with Optuna (50 trials each) — XGBoost won at ROC-AUC 0.5916
- Confidence threshold analysis — at 0.60 threshold accuracy jumps to 62.66% on 37.7% of trades
- SHAP beeswarm plots show direction and magnitude of each feature's impact on predictions
- Dynamic model saving — winner auto-named based on algorithm (e.g. `xgboost_tuned.pkl`)
- 46 features including interaction terms: RSI × volume spike, MACD × sentiment, z-score × RSI

**Quant Finance**
- Monte Carlo validation of Black-Scholes pricer against real options chain prices
- Portfolio optimization conditioned on ML signal — only optimizes over top predicted tickers
- Regime-aware strategy: only deploy capital in bull regimes detected by HMM

---

*Built by Adi0015 · April 2026*
