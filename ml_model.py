import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import shap
import optuna
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (accuracy_score, f1_score,
                                     roc_auc_score, classification_report,
                                     confusion_matrix)
from sklearn.preprocessing   import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Feature columns ───────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Trend
    "ema_12", "ema_26", "macd", "macd_signal", "macd_diff",
    "ema_50", "ema_200", "above_ema50", "above_ema200", "ema_50_200_cross",
    # Momentum
    "rsi", "roc_10", "williams_r", "rsi_ma10", "rsi_divergence",
    # Volatility
    "bb_pct", "bb_width", "atr", "realized_vol_20", "zscore_20",
    # Volume
    "obv", "vol_ratio", "vol_spike",
    # Price derived
    "return_5d", "return_20d", "dist_52w_high", "dist_52w_low",
    "overnight_gap", "price_vs_spy", "daily_return", "log_return",
    # Macro
    "cpi", "fed_rate", "yield_10y", "yield_2y",
    "yield_spread", "cpi_mom", "unemployment",
    # Sentiment
    "sentiment_mean", "sentiment_std", "headline_count",
    # Interactions
    "rsi_vol_spike", "macd_sentiment", "vol_momentum",
    "zscore_rsi", "macro_sentiment",
]

TARGET_COL = "target"
TEST_SIZE  = 0.2


# ── Load data ─────────────────────────────────────────────────────────────────

def load_data(path="data/final_features.parquet"):
    print("\n[1/6] Loading data...")
    df = pd.read_parquet(path)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Only keep feature cols that actually exist
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"    WARN: missing columns (will skip): {missing}")

    cols_needed = available + [TARGET_COL]
    df = df.dropna(subset=cols_needed)

    print(f"    Shape          : {df.shape}")
    print(f"    Features used  : {len(available)}")
    print(f"    Date range     : {df['date'].min()} → {df['date'].max()}")
    print(f"    Target balance : {df[TARGET_COL].value_counts().to_dict()}")
    return df, available


# ── Time-series split ─────────────────────────────────────────────────────────

def time_series_split(df, feature_cols):
    print("\n[2/6] Time-series split (no shuffling)...")
    dates      = sorted(df["date"].unique())
    cutoff_idx = int(len(dates) * (1 - TEST_SIZE))
    cutoff     = dates[cutoff_idx]

    train = df[df["date"] <  cutoff]
    test  = df[df["date"] >= cutoff]

    print(f"    Cutoff : {cutoff}")
    print(f"    Train  : {train.shape} | Test: {test.shape}")

    return (train[feature_cols], test[feature_cols],
            train[TARGET_COL],   test[TARGET_COL])


# ── Evaluate helper ───────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, scaler=None):
    X = scaler.transform(X_test) if scaler else X_test
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "F1"      : round(f1_score(y_test, preds), 4),
        "ROC-AUC" : round(roc_auc_score(y_test, proba), 4),
    }


# ── Optuna tuning for XGBoost ─────────────────────────────────────────────────

def tune_xgboost(X_train, X_test, y_train, y_test, n_trials=50):
    print("\n[3/6] Tuning XGBoost with Optuna (50 trials)...")

    def objective(trial):
        params = {
            "n_estimators"        : trial.suggest_int("n_estimators", 200, 1000),
            "max_depth"           : trial.suggest_int("max_depth", 3, 7),
            "learning_rate"       : trial.suggest_float("learning_rate", 0.01, 0.15),
            "subsample"           : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree"    : trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight"    : trial.suggest_int("min_child_weight", 1, 10),
            "gamma"               : trial.suggest_float("gamma", 0, 0.5),
            "reg_alpha"           : trial.suggest_float("reg_alpha", 0, 1.0),
            "reg_lambda"          : trial.suggest_float("reg_lambda", 0.5, 2.0),
            "eval_metric"         : "logloss",
            "random_state"        : 42,
            "verbosity"           : 0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
        proba = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"    Best ROC-AUC : {study.best_value:.4f}")
    print(f"    Best params  : {study.best_params}")

    best_model = xgb.XGBClassifier(
        **study.best_params,
        eval_metric  = "logloss",
        random_state = 42,
        verbosity    = 0,
    )
    best_model.fit(X_train, y_train,
                   eval_set=[(X_test, y_test)],
                   verbose=False)
    return best_model, study.best_params


# ── Train all models ──────────────────────────────────────────────────────────

def train_all_models(X_train, X_test, y_train, y_test):
    print("\n[4/6] Training all models for comparison...")
    results = {}

    # 1. Logistic Regression
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    lr      = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr_sc, y_train)
    results["Logistic Regression"] = evaluate(lr, X_test, y_test, scaler)

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=6,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["Random Forest"] = evaluate(rf, X_test, y_test)

    # 3. LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set  = [(X_test, y_test)],
        callbacks = [lgb.early_stopping(30, verbose=False),
                     lgb.log_evaluation(period=-1)],
    )
    results["LightGBM"] = evaluate(lgb_model, X_test, y_test)

    return results, scaler, lgb_model


# ── Confidence threshold analysis ────────────────────────────────────────────

def confidence_analysis(model, X_test, y_test):
    print("\n[5/6] Confidence threshold analysis...")
    proba = model.predict_proba(X_test)[:, 1]
    df    = pd.DataFrame({"proba": proba, "actual": y_test.values})

    print(f"\n    {'Threshold':>12} {'Trades':>8} {'Coverage':>10} {'Accuracy':>10}")
    print("    " + "-" * 44)
    rows = []
    for threshold in [0.50, 0.52, 0.55, 0.58, 0.60, 0.63, 0.65]:
        mask   = (df["proba"] > threshold) | (df["proba"] < 1 - threshold)
        subset = df[mask]
        if len(subset) < 10:
            continue
        preds    = (subset["proba"] > 0.5).astype(int)
        acc      = accuracy_score(subset["actual"], preds)
        coverage = len(subset) / len(df)
        print(f"    {threshold:>12.2f} {len(subset):>8} "
              f"{coverage:>10.1%} {acc:>10.4f}")
        rows.append({"threshold": threshold, "trades": len(subset),
                     "coverage": coverage, "accuracy": acc})
    return pd.DataFrame(rows)


# ── SHAP explanations ─────────────────────────────────────────────────────────

def explain_model(model, X_test, feature_cols):
    print("\n[6/6] Generating SHAP plots...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar",
                      show=False, max_display=20)
    plt.title("Feature Importance — Mean |SHAP|")
    plt.tight_layout()
    plt.savefig("notebooks/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=20)
    plt.title("SHAP Beeswarm — Feature Impact Direction")
    plt.tight_layout()
    plt.savefig("notebooks/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("    Saved: notebooks/shap_importance.png")
    print("    Saved: notebooks/shap_beeswarm.png")
    return shap_values


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  AlphaLens — Day 3: ML Return Predictor (Enhanced)")
    print("=" * 55)

    df, feature_cols = load_data()
    X_train, X_test, y_train, y_test = time_series_split(df, feature_cols)

    # Tune XGBoost with Optuna
    xgb_model, best_params = tune_xgboost(X_train, X_test,
                                           y_train, y_test, n_trials=50)

    # Compare all models
    results, scaler, lgb_model = train_all_models(X_train, X_test,
                                                   y_train, y_test)

    # Add tuned XGBoost to results
    results["XGBoost (tuned)"] = evaluate(xgb_model, X_test, y_test)

    # Print comparison table
    comparison = pd.DataFrame(results).T.sort_values("ROC-AUC", ascending=False)
    print(f"\n    {'Model':<25} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("    " + "-" * 58)
    for name, row in comparison.iterrows():
        print(f"    {name:<25} {row['Accuracy']:>10.4f} "
              f"{row['F1']:>10.4f} {row['ROC-AUC']:>10.4f}")

    comparison.to_csv("notebooks/model_comparison.csv")

    # Confidence analysis on best model
    conf_df = confidence_analysis(xgb_model, X_test, y_test)
    conf_df.to_csv("notebooks/confidence_analysis.csv", index=False)

    # SHAP on best model
    explain_model(xgb_model, X_test, feature_cols)

    # Save best model
    joblib.dump(xgb_model, "models/xgboost_model.pkl")
    joblib.dump(scaler,    "models/scaler.pkl")
    joblib.dump(best_params, "models/best_params.pkl")
    print("\n    Saved: models/xgboost_model.pkl")
    print("    Saved: models/best_params.pkl")

    best = comparison.index[0]
    print("\n" + "=" * 55)
    print(f"  Best model    : {best}")
    print(f"  ROC-AUC       : {comparison.loc[best, 'ROC-AUC']}")
    print(f"  Accuracy      : {comparison.loc[best, 'Accuracy']}")
    print("=" * 55)
    print("\nDay 3 complete — ready for Day 4!")
