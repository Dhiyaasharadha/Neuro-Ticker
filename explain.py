"""
explain.py
----------
Uses SHAP (TreeExplainer) to compute feature importance / attributions for
the trained XGBoost model's most recent prediction, so users can see which
features (RSI, sentiment, moving averages, volatility, ...) drove the
forecast.

Usage:
    python explain.py --model data/xgb_model.json --features data/features_complete.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

FEATURE_COLS = ["close", "ma_7", "ma_30", "vol_7", "rsi_14", "avg_sentiment"]


def load_xgb_model(path: str):
    from xgboost import XGBRegressor

    model = XGBRegressor()
    model.load_model(path)
    return model


def compute_shap_for_latest(model, features_df: pd.DataFrame, background_rows: int = 200):
    import shap

    X_background = features_df[FEATURE_COLS].tail(background_rows)
    X_latest = features_df[FEATURE_COLS].tail(1)

    explainer = shap.TreeExplainer(model, X_background)
    shap_values = explainer.shap_values(X_latest)

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.array(base_value).flatten()[0])

    contributions = {
        col: float(shap_values[0][i]) for i, col in enumerate(FEATURE_COLS)
    }
    return contributions, float(base_value)


def compute_shap_matrix(model, features_df: pd.DataFrame, n_rows: int = 100):
    """SHAP values across the last n_rows, useful for a summary bar chart."""
    import shap

    X = features_df[FEATURE_COLS].tail(n_rows)
    explainer = shap.TreeExplainer(model, X)
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return {col: float(mean_abs[i]) for i, col in enumerate(FEATURE_COLS)}


def main():
    parser = argparse.ArgumentParser(description="Compute SHAP explanations for the XGBoost forecast.")
    parser.add_argument("--model", default="data/xgb_model.json")
    parser.add_argument("--features", default="data/features_complete.csv")
    parser.add_argument("--out", default="data/shap_explanation.json")
    args = parser.parse_args()

    print(f"[explain] Loading XGBoost model from {args.model}")
    model = load_xgb_model(args.model)

    print(f"[explain] Loading features from {args.features}")
    features_df = pd.read_csv(args.features, parse_dates=["date"])

    print("[explain] Computing SHAP values for the latest prediction ...")
    contributions, base_value = compute_shap_for_latest(model, features_df)

    print("[explain] Computing global mean |SHAP| feature importance (last 100 rows) ...")
    global_importance = compute_shap_matrix(model, features_df)

    result = {
        "base_value": base_value,
        "latest_contributions": contributions,
        "global_importance_mean_abs": global_importance,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[explain] Saved SHAP explanation -> {args.out}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
