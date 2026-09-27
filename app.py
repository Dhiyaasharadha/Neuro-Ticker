"""
app.py
------
NeuroTicker — Explainable Multi-Model Stock Intelligence Platform
Interactive Streamlit dashboard.

Run with:
    streamlit run app.py
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = "data"

st.set_page_config(page_title="NeuroTicker", layout="wide", page_icon="📈")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["date"]) if "date" in open(path).readline() else pd.read_csv(path)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def run_pipeline(ticker: str, start_date: str):
    """Re-run the full data -> feature -> sentiment -> model pipeline for a given ticker."""
    steps = [
        [sys.executable, "data_collection.py", "--ticker", ticker, "--start", start_date],
        [sys.executable, "features.py"],
        [sys.executable, "sentiment.py", "--ticker", ticker, "--company", ticker, "--start", start_date],
        [sys.executable, "models.py"],
        [sys.executable, "explain.py"],
    ]
    logs = []
    for step in steps:
        result = subprocess.run(step, capture_output=True, text=True)
        logs.append((", ".join(step), result.returncode, result.stdout[-1500:], result.stderr[-1500:]))
        if result.returncode != 0:
            return logs, False
    return logs, True


def xgb_predict_with_sentiment(feature_row: dict, sentiment_override: float):
    """Reload the saved XGBoost model and predict with a manually overridden
    sentiment value, for the live what-if slider (does not require retraining)."""
    from xgboost import XGBRegressor

    model = XGBRegressor()
    model.load_model(os.path.join(DATA_DIR, "xgb_model.json"))

    cols = ["close", "ma_7", "ma_30", "vol_7", "rsi_14", "avg_sentiment"]
    row = feature_row.copy()
    row["avg_sentiment"] = sentiment_override
    X = np.array([[row[c] for c in cols]])
    return float(model.predict(X)[0])


# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.title("📈 NeuroTicker")
st.sidebar.caption("Explainable Multi-Model Stock Intelligence Platform")

ticker = st.sidebar.text_input("Ticker symbol", value="AAPL").upper().strip()
start_date = st.sidebar.date_input("History start date", value=pd.to_datetime("2015-01-01"))

if st.sidebar.button("🔄 Run / refresh pipeline for this ticker", type="primary"):
    with st.spinner(f"Running full pipeline for {ticker} ..."):
        logs, ok = run_pipeline(ticker, str(start_date))
    st.cache_data.clear()
    if ok:
        st.sidebar.success("Pipeline completed successfully.")
    else:
        st.sidebar.error("Pipeline failed — see details below.")
    with st.sidebar.expander("Pipeline logs"):
        for name, code, out, err in logs:
            st.write(f"**{name}** (exit {code})")
            if out:
                st.code(out)
            if err:
                st.code(err)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Note: in network-restricted environments, data_collection.py and "
    "sentiment.py automatically fall back to clearly-labeled synthetic "
    "data so the dashboard always has something to show."
)

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
features_path = os.path.join(DATA_DIR, "features_complete.csv")
forecast_path = os.path.join(DATA_DIR, "forecast.json")
shap_path = os.path.join(DATA_DIR, "shap_explanation.json")
raw_prices_path = os.path.join(DATA_DIR, "raw_prices.csv")

features_df = load_csv(features_path)
forecast = load_json(forecast_path)
shap_data = load_json(shap_path)
raw_df = load_csv(raw_prices_path)

st.title("NeuroTicker — Explainable Multi-Model Stock Intelligence")

if features_df is None or forecast is None:
    st.warning(
        "No data found yet. Click **'Run / refresh pipeline for this ticker'** "
        "in the sidebar to download data, engineer features, score sentiment, "
        "train models, and generate a forecast."
    )
    st.stop()

data_source_note = raw_df["data_source"].iloc[-1] if raw_df is not None and "data_source" in raw_df.columns else "unknown"
if "synthetic" in str(data_source_note):
    st.info(
        f"⚠️ Price data source: **{data_source_note}**. Real Yahoo Finance data could not be "
        f"reached from this environment, so a realistic synthetic series is being used so the "
        f"full pipeline can be demonstrated end-to-end. Run this app on a machine with normal "
        f"internet access to get real {ticker} data automatically."
    )

# ----------------------------------------------------------------------
# Row 1: Price chart + forecast summary
# ----------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"{forecast.get('ticker', ticker)} — Historical Close Price")
    plot_df = features_df.tail(250)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["close"], name="Close", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["ma_7"], name="MA 7", line=dict(color="orange", dash="dot")))
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["ma_30"], name="MA 30", line=dict(color="green", dash="dot")))

    # Add the forecast point + CI as an extra marker after the last date
    pred_date = pd.to_datetime(forecast["predicted_for_date"])
    ci = forecast["confidence_interval_90"]
    fig.add_trace(go.Scatter(
        x=[pred_date], y=[forecast["predictions"]["ensemble"]],
        mode="markers", name="Ensemble forecast",
        marker=dict(color="red", size=12, symbol="star"),
    ))
    fig.add_trace(go.Scatter(
        x=[pred_date, pred_date], y=[ci["lower"], ci["upper"]],
        mode="lines", name="90% CI", line=dict(color="red", width=4),
        opacity=0.4,
    ))
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Next-Day Forecast")
    preds = forecast["predictions"]
    st.metric(
        label=f"Ensemble prediction for {forecast['predicted_for_date']}",
        value=f"${preds['ensemble']:.2f}",
        delta=f"{preds['ensemble'] - forecast['current_close']:+.2f} vs last close",
    )
    st.caption(f"90% confidence interval: **${ci['lower']:.2f} – ${ci['upper']:.2f}**")

    st.markdown("**Individual model predictions**")
    st.table(pd.DataFrame({
        "Model": ["LSTM", "XGBoost", "Prophet", "Ensemble"],
        "Predicted price": [f"${preds['lstm']:.2f}", f"${preds['xgboost']:.2f}",
                             f"${preds['prophet']:.2f}", f"${preds['ensemble']:.2f}"],
    }))

    sentiment_val = forecast["current_sentiment"]
    sentiment_label = "Positive 🙂" if sentiment_val > 0.05 else "Negative 🙁" if sentiment_val < -0.05 else "Neutral 😐"
    st.markdown("**News sentiment (FinBERT)**")
    st.progress(min(max((sentiment_val + 1) / 2, 0.0), 1.0))
    st.caption(f"Current avg. sentiment score: **{sentiment_val:+.3f}** ({sentiment_label})")

# ----------------------------------------------------------------------
# Row 2: SHAP explainability
# ----------------------------------------------------------------------
st.markdown("---")
st.subheader("🔍 Why this prediction? (SHAP feature attribution — XGBoost)")

if shap_data:
    contrib = shap_data["latest_contributions"]
    contrib_df = pd.DataFrame({"feature": list(contrib.keys()), "shap_value": list(contrib.values())})
    contrib_df = contrib_df.sort_values("shap_value")

    fig_shap = go.Figure(go.Bar(
        x=contrib_df["shap_value"], y=contrib_df["feature"], orientation="h",
        marker_color=["#d62728" if v < 0 else "#2ca02c" for v in contrib_df["shap_value"]],
    ))
    fig_shap.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="SHAP value (impact on predicted price, $)",
        title=f"Base value: ${shap_data['base_value']:.2f}",
    )
    st.plotly_chart(fig_shap, use_container_width=True)
    st.caption(
        "Green bars push the prediction **up**, red bars push it **down**, relative to the "
        "model's base (average) prediction. Bar length = magnitude of that feature's influence "
        "on today's forecast."
    )
else:
    st.info("Run the pipeline to generate SHAP explanations.")

# ----------------------------------------------------------------------
# Row 3: What-if sentiment slider
# ----------------------------------------------------------------------
st.markdown("---")
st.subheader("🎛️ What-if: adjust sentiment and see the price update live")

current_features = forecast["current_features"]
default_sentiment = float(current_features["avg_sentiment"])

whatif_sentiment = st.slider(
    "Manually set news sentiment score (-1 = very negative, +1 = very positive)",
    min_value=-1.0, max_value=1.0, value=float(np.clip(default_sentiment, -1.0, 1.0)), step=0.01,
)

if os.path.exists(os.path.join(DATA_DIR, "xgb_model.json")):
    whatif_pred = xgb_predict_with_sentiment(current_features, whatif_sentiment)
    baseline_pred = xgb_predict_with_sentiment(current_features, default_sentiment)

    wcol1, wcol2, wcol3 = st.columns(3)
    wcol1.metric("Baseline sentiment", f"{default_sentiment:+.3f}")
    wcol2.metric("What-if sentiment", f"{whatif_sentiment:+.3f}")
    wcol3.metric(
        "XGBoost price under what-if sentiment",
        f"${whatif_pred:.2f}",
        delta=f"{whatif_pred - baseline_pred:+.2f} vs baseline",
    )

    st.caption(
        "This slider re-runs the already-trained XGBoost model instantly with a modified "
        "sentiment input — no retraining needed — so you can explore 'what would the model "
        "predict if news sentiment were more positive/negative?'"
    )
else:
    st.info("Run the pipeline first to enable the what-if slider.")

st.markdown("---")
st.caption(
    "NeuroTicker blends an LSTM, XGBoost, and Prophet model with FinBERT news sentiment. "
    "This is a research/educational tool, not investment advice."
)
