"""
models.py
---------
Trains three forecasting models on the engineered feature set and blends
their predictions into a single ensemble next-day price forecast, with a
bootstrap-based confidence interval.

Models:
    1. LSTM (TensorFlow/Keras)   - sequence model over the last N days of features
    2. XGBoost regressor          - tabular model over the latest feature row
    3. Prophet                    - univariate time series model on close price

Ensemble:
    ensemble_pred = mean(lstm_pred, xgb_pred, prophet_pred)

Confidence interval:
    Bootstrap resampling of historical residuals (out-of-sample one-step-ahead
    errors) added to the point forecast, giving an empirical 90% interval.

Usage:
    python models.py --features data/features.csv --sentiment data/sentiment.csv
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FEATURE_COLS = ["close", "ma_7", "ma_30", "vol_7", "rsi_14", "avg_sentiment"]
SEQ_LEN = 10  # LSTM lookback window


# ----------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------
def load_data(features_path: str, sentiment_path: str) -> pd.DataFrame:
    feat = pd.read_csv(features_path, parse_dates=["date"])
    sent = pd.read_csv(sentiment_path, parse_dates=["date"])
    sent = sent[["date", "avg_sentiment"]]

    df = feat.merge(sent, on="date", how="left")
    df["avg_sentiment"] = df["avg_sentiment"].ffill().bfill().fillna(0.0)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def train_test_split_tail(df: pd.DataFrame, n_test: int = 60):
    train = df.iloc[:-n_test].copy()
    test = df.iloc[-n_test:].copy()
    return train, test


# ----------------------------------------------------------------------
# XGBoost
# ----------------------------------------------------------------------
def train_xgboost(train_df: pd.DataFrame):
    from xgboost import XGBRegressor

    X = train_df[FEATURE_COLS].values
    y = train_df["target_next_close"].values
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]

    model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X, y)
    return model


def predict_xgboost(model, row: pd.DataFrame):
    X = row[FEATURE_COLS].values
    return float(model.predict(X)[0])


# ----------------------------------------------------------------------
# LSTM
# ----------------------------------------------------------------------
def make_sequences(values: np.ndarray, targets: np.ndarray, seq_len: int):
    xs, ys = [], []
    for i in range(seq_len, len(values)):
        xs.append(values[i - seq_len:i])
        ys.append(targets[i])
    return np.array(xs), np.array(ys)


def train_lstm(train_df: pd.DataFrame, seq_len: int = SEQ_LEN):
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    feats = train_df[FEATURE_COLS].values
    targets = train_df["target_next_close"].values

    valid = ~np.isnan(targets)
    feats, targets = feats[valid], targets[valid]

    scaler_x = StandardScaler().fit(feats)
    feats_scaled = scaler_x.transform(feats)

    y_mean, y_std = targets.mean(), targets.std() + 1e-8
    targets_scaled = (targets - y_mean) / y_std

    X, y = make_sequences(feats_scaled, targets_scaled, seq_len)

    model = Sequential([
        LSTM(32, input_shape=(seq_len, len(FEATURE_COLS)), return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=15, batch_size=16, verbose=0)

    return model, scaler_x, y_mean, y_std


def predict_lstm(model, scaler_x, y_mean, y_std, recent_df: pd.DataFrame, seq_len: int = SEQ_LEN):
    feats = recent_df[FEATURE_COLS].values[-seq_len:]
    feats_scaled = scaler_x.transform(feats)
    X = feats_scaled.reshape(1, seq_len, len(FEATURE_COLS))
    pred_scaled = model.predict(X, verbose=0)[0][0]
    return float(pred_scaled * y_std + y_mean)


# ----------------------------------------------------------------------
# Prophet
# ----------------------------------------------------------------------
def train_prophet(train_df: pd.DataFrame):
    from prophet import Prophet

    p_df = train_df[["date", "close"]].rename(columns={"date": "ds", "close": "y"})
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(p_df)
    return model


def predict_prophet(model, last_date: pd.Timestamp):
    future = pd.DataFrame({"ds": [last_date + pd.Timedelta(days=1)]})
    forecast = model.predict(future)
    return float(forecast["yhat"].iloc[0])


# ----------------------------------------------------------------------
# Bootstrap confidence interval
# ----------------------------------------------------------------------
def bootstrap_confidence_interval(residuals: np.ndarray, point_estimate: float,
                                   n_boot: int = 2000, ci: float = 0.90, seed: int = 42):
    """
    Build an empirical CI by bootstrap-resampling historical one-step-ahead
    residuals and adding them to the point forecast.
    """
    rng = np.random.default_rng(seed)
    if len(residuals) == 0:
        residuals = np.array([0.0])

    boot_means = np.array([
        rng.choice(residuals, size=len(residuals), replace=True).mean()
        for _ in range(n_boot)
    ])
    boot_preds = point_estimate + boot_means

    alpha = 1 - ci
    lower = float(np.quantile(boot_preds, alpha / 2))
    upper = float(np.quantile(boot_preds, 1 - alpha / 2))
    return lower, upper


def compute_backtest_residuals(train_df: pd.DataFrame, xgb_model) -> np.ndarray:
    """One-step-ahead XGBoost residuals over the training set tail, used to
    characterize typical forecast error for the bootstrap CI."""
    tail = train_df.tail(200).copy()
    y_true = tail["target_next_close"].values
    valid = ~np.isnan(y_true)
    X = tail[FEATURE_COLS].values[valid]
    y_true = y_true[valid]
    y_pred = xgb_model.predict(X)
    return y_true - y_pred


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def run_pipeline(features_path: str, sentiment_path: str, out_path: str, n_test: int = 60):
    print("[models] Loading merged feature + sentiment data ...")
    df = load_data(features_path, sentiment_path)
    df_complete = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    train_df, _ = train_test_split_tail(df_complete, n_test=n_test)

    print(f"[models] Training on {len(train_df)} rows (holding out last {n_test} for backtest residuals) ...")

    print("[models] Training XGBoost ...")
    xgb_model = train_xgboost(train_df)

    print("[models] Training LSTM (TensorFlow/Keras) ...")
    lstm_model, scaler_x, y_mean, y_std = train_lstm(train_df)

    print("[models] Training Prophet ...")
    prophet_model = train_prophet(train_df)

    # Predict for "tomorrow" using the most recent complete row of features
    last_row = df_complete.iloc[[-1]]
    last_date = pd.to_datetime(df_complete["date"].iloc[-1])

    xgb_pred = predict_xgboost(xgb_model, last_row)
    lstm_pred = predict_lstm(lstm_model, scaler_x, y_mean, y_std, df_complete)
    prophet_pred = predict_prophet(prophet_model, last_date)

    ensemble_pred = float(np.mean([xgb_pred, lstm_pred, prophet_pred]))

    residuals = compute_backtest_residuals(train_df, xgb_model)
    lower, upper = bootstrap_confidence_interval(residuals, ensemble_pred)

    result = {
        "ticker": df_complete["ticker"].iloc[-1] if "ticker" in df_complete.columns else None,
        "as_of_date": str(last_date.date()),
        "predicted_for_date": str((last_date + pd.Timedelta(days=1)).date()),
        "current_close": float(last_row["close"].iloc[0]),
        "predictions": {
            "xgboost": xgb_pred,
            "lstm": lstm_pred,
            "prophet": prophet_pred,
            "ensemble": ensemble_pred,
        },
        "confidence_interval_90": {"lower": lower, "upper": upper},
        "current_sentiment": float(last_row["avg_sentiment"].iloc[0]),
        "current_features": {c: float(last_row[c].iloc[0]) for c in FEATURE_COLS},
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Also persist the trained xgboost model + training data reference for explain.py
    xgb_model.save_model(os.path.join(os.path.dirname(out_path), "xgb_model.json"))
    train_df.to_csv(os.path.join(os.path.dirname(out_path), "train_snapshot.csv"), index=False)
    df_complete.to_csv(os.path.join(os.path.dirname(out_path), "features_complete.csv"), index=False)

    print(f"[models] Saved forecast -> {out_path}")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Train ensemble forecasting models.")
    parser.add_argument("--features", default="data/features.csv")
    parser.add_argument("--sentiment", default="data/sentiment.csv")
    parser.add_argument("--out", default="data/forecast.json")
    parser.add_argument("--n-test", type=int, default=60)
    args = parser.parse_args()

    run_pipeline(args.features, args.sentiment, args.out, n_test=args.n_test)


if __name__ == "__main__":
    main()
