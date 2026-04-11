import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

load_dotenv()

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM",
           "GS", "SPY", "QQQ", "TSLA", "NVDA"]

TICKER_NAMES = {
    "AAPL": "Apple",       "MSFT": "Microsoft", "GOOGL": "Google",
    "AMZN": "Amazon",      "JPM":  "JPMorgan",  "GS":    "Goldman Sachs",
    "SPY":  "S&P 500",     "QQQ":  "Nasdaq",    "TSLA":  "Tesla",
    "NVDA": "Nvidia",
}


def load_finbert():
    print("\n[1/3] Loading FinBERT (~500MB, downloads once)...")
    model_name = "ProsusAI/finbert"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    print("    FinBERT ready.")
    return tokenizer, model


def score_headlines(headlines, tokenizer, model, batch_size=16):
    """Score = P(positive) - P(negative), range [-1, +1]"""
    if not headlines:
        return []
    scores = []
    for i in range(0, len(headlines), batch_size):
        batch  = headlines[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs  = F.softmax(logits, dim=-1).numpy()
        # FinBERT label order: positive=0, negative=1, neutral=2
        scores.extend((probs[:, 0] - probs[:, 1]).tolist())
    return scores


def fetch_and_score(tickers, tokenizer, model):
    print("\n[2/3] Fetching headlines and scoring sentiment...")
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        raise ValueError("NEWS_API_KEY missing from .env")

    newsapi = NewsApiClient(api_key=api_key)
    records = []

    for ticker in tqdm(tickers, desc="Scoring"):
        query = TICKER_NAMES.get(ticker, ticker)
        try:
            resp     = newsapi.get_everything(q=query, language="en",
                                              sort_by="publishedAt", page_size=100)
            articles = resp.get("articles", [])
        except Exception as e:
            print(f"\n    WARN: {ticker} failed — {e}")
            continue

        if not articles:
            continue

        headlines = [a["title"] for a in articles if a.get("title")]
        dates     = [a["publishedAt"][:10] for a in articles if a.get("title")]
        scores    = score_headlines(headlines, tokenizer, model)

        for date, score in zip(dates, scores):
            records.append({"date": date, "ticker": ticker, "sentiment": score})

        time.sleep(0.25)

    if not records:
        print("    WARNING: No records — check your NewsAPI key")
        return pd.DataFrame(columns=["date", "ticker", "sentiment_mean",
                                     "sentiment_std", "headline_count"])

    raw = pd.DataFrame(records)
    raw["date"] = pd.to_datetime(raw["date"])

    agg = (raw.groupby(["date", "ticker"])["sentiment"]
              .agg(sentiment_mean="mean",
                   sentiment_std="std",
                   headline_count="count")
              .reset_index())
    agg["sentiment_std"] = agg["sentiment_std"].fillna(0)
    print(f"    Collected {len(agg)} date×ticker sentiment records")
    return agg


def merge_sentiment(features_df, sentiment_df):
    print("\n[3/3] Merging sentiment into features...")
    features_df["date"]  = pd.to_datetime(features_df["date"]).dt.normalize()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.normalize()

    merged = pd.merge(features_df, sentiment_df,
                      on=["date", "ticker"], how="left")

    for col in ["sentiment_mean", "sentiment_std", "headline_count"]:
        merged[col] = merged.groupby("ticker")[col].ffill().fillna(0)

    print(f"    Final shape: {merged.shape}")
    print(f"    Sentiment coverage: "
          f"{(merged['sentiment_mean'] != 0).mean() * 100:.1f}% of rows")
    return merged


if __name__ == "__main__":
    print("=" * 50)
    print("  AlphaLens — Day 2b: Sentiment Engine")
    print("=" * 50)

    tokenizer, model = load_finbert()
    features_df      = pd.read_parquet("data/features_data.parquet")
    sentiment_df     = fetch_and_score(TICKERS, tokenizer, model)

    sentiment_df.to_parquet("data/sentiment_raw.parquet", index=False)
    print(f"Saved: data/sentiment_raw.parquet")

    final_df = merge_sentiment(features_df, sentiment_df)
    final_df.to_parquet("data/final_features.parquet", index=False)
    print(f"Saved: data/final_features.parquet {final_df.shape}")
    print("\nDay 2 complete — ready for Day 3!")
