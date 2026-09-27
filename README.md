# NeuroTicker — Explainable Multi-Model Stock Intelligence Platform

NeuroTicker forecasts next-day stock prices by blending three different
model families — an **LSTM** (sequence model), **XGBoost** (gradient-boosted
trees), and **Prophet** (Bayesian time-series decomposition) — with a
**FinBERT** news-sentiment signal, then explains *why* it made that
prediction using **SHAP** feature attribution. Everything is viewable and
interactive in a **Streamlit** dashboard, including a live "what-if" slider
that lets you manually change the sentiment input and instantly see how the
model's prediction reacts.

Default ticker is **AAPL**, but the ticker is fully configurable from the
sidebar (or CLI flags on each script).

---

## Architecture

```
                ┌─────────────────────┐
                │  data_collection.py  │  yfinance -> data/raw_prices.csv
                └──────────┬───────────┘
                           │
                ┌──────────▼───────────┐
                │     features.py      │  MA-7, MA-30, 7d volatility, RSI-14
                └──────────┬───────────┘  -> data/features.csv
                           │
   ┌───────────────────────┼───────────────────────┐
   │                       │                       │
┌──▼───────────┐  ┌────────▼────────┐               │
│ sentiment.py │  │                 │               │
│ RSS headlines│  │                 │               │
│  + FinBERT   │  │                 │               │
└──────┬───────┘  │                 │               │
       │ data/sentiment.csv         │               │
       └───────────────►┌───────────▼──────────┐    │
                         │      models.py        │◄──┘
                         │  LSTM + XGBoost +      │
                         │  Prophet -> ensemble   │
                         │  + bootstrap 90% CI    │
                         └───────────┬────────────┘
                                     │ data/forecast.json
                                     │ data/xgb_model.json
                        ┌────────────▼────────────┐
                        │       explain.py         │
                        │  SHAP TreeExplainer on   │
                        │  the XGBoost model        │
                        └────────────┬─────────────┘
                                     │ data/shap_explanation.json
                        ┌────────────▼─────────────┐
                        │          app.py           │
                        │   Streamlit dashboard:     │
                        │  price chart, forecast+CI, │
                        │  sentiment gauge, SHAP     │
                        │  chart, what-if slider     │
                        └────────────────────────────┘
```

## Pipeline stages

| File | Purpose | Output |
|---|---|---|
| `data_collection.py` | Downloads daily OHLCV history via `yfinance` | `data/raw_prices.csv` |
| `features.py` | Computes MA-7, MA-30, 7-day rolling volatility, 14-day RSI, and the next-day-close training target | `data/features.csv` |
| `sentiment.py` | Pulls recent headlines from free RSS feeds (Yahoo Finance / Google News, no API key) and scores each with pretrained **FinBERT** (`ProsusAI/finbert`); aggregates to a daily average sentiment score | `data/sentiment.csv`, `data/sentiment_headlines.csv` |
| `models.py` | Trains an LSTM (Keras), XGBoost, and Prophet on the merged feature+sentiment table; blends the three point forecasts into an ensemble; builds a 90% confidence interval via bootstrap resampling of historical XGBoost residuals | `data/forecast.json`, `data/xgb_model.json` |
| `explain.py` | Runs SHAP `TreeExplainer` on the trained XGBoost model to attribute the latest prediction (and global importance) to each input feature | `data/shap_explanation.json` |
| `app.py` | Streamlit dashboard tying it all together, including a live what-if sentiment slider | interactive web app |

## The ensemble & confidence interval

```
ensemble_prediction = mean(LSTM_pred, XGBoost_pred, Prophet_pred)
```

The 90% confidence interval is built by **bootstrap resampling** the
XGBoost model's last 200 one-step-ahead residuals 2,000 times, adding the
resampled mean residual to the ensemble point estimate, and taking the
5th/95th percentiles of the resulting distribution. This is an empirical,
data-driven interval rather than a single point guess.

## Explainability

`explain.py` uses SHAP's `TreeExplainer` (exact, fast Shapley values for
tree ensembles) on the XGBoost model to show, for the most recent
prediction: how much each feature (last close price, MA-7, MA-30, 7-day
volatility, RSI-14, average sentiment) pushed the prediction up or down
relative to the model's average ("base") prediction. The dashboard renders
this as a horizontal bar chart (green = pushes price up, red = pushes it
down).

## The what-if sentiment slider

The dashboard's "what-if" slider re-runs the **already-trained** XGBoost
model with every other feature held constant and only `avg_sentiment`
swapped for whatever value you drag the slider to (from -1 = very negative
to +1 = very positive). No retraining happens — this is instant. It shows
you, in isolation, how sensitive the model's prediction currently is to
news sentiment.

---

## Setup & installation

```bash
# from the project folder
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions that were tested end-to-end
while building this project (Python 3.12). Notable version choices:

- **torch 2.3.1** and **transformers 4.44.2** are pinned together
  deliberately — transformers 5.x requires torch ≥ 2.5, and the newest
  torch releases pull in `nvidia-*-cu13` CUDA packages via PyPI that can
  be flaky/huge to install on a CPU-only machine. This combination is
  stable and CPU-friendly.
- **tensorflow-cpu** is used instead of the full `tensorflow` package to
  avoid pulling in unnecessary GPU dependencies.

If you have a CUDA-capable GPU and want GPU acceleration for torch/
tensorflow, install the GPU-enabled variants instead (see each library's
docs) — everything in this codebase runs fine on CPU too, just slower.

## Running the full pipeline

Run each stage once, in order, the first time (or whenever you want to
refresh the data/models):

```bash
python data_collection.py --ticker AAPL --start 2015-01-01
python features.py
python sentiment.py --ticker AAPL --company "Apple" --start 2015-01-01
python models.py
python explain.py
```

Then launch the dashboard:

```bash
streamlit run app.py
```

It will open at **http://localhost:8501**. You can also trigger the whole
pipeline for any ticker directly from the sidebar's "Run / refresh
pipeline for this ticker" button — just change the ticker symbol and start
date first.

### Changing the ticker

Every script accepts `--ticker` (and `sentiment.py` also accepts
`--company`, since headlines are searched by company name). For example,
to switch to Microsoft:

```bash
python data_collection.py --ticker MSFT --start 2018-01-01
python features.py
python sentiment.py --ticker MSFT --company "Microsoft" --start 2018-01-01
python models.py
python explain.py
```

Or just type `MSFT` into the sidebar and click the refresh button.

## Important note on data sources & offline fallbacks

`data_collection.py` and `sentiment.py` are written to use **real, free
data sources with no paid API keys**: `yfinance` for prices, and free RSS
feeds (Yahoo Finance headline RSS, Google News RSS) plus pretrained
**FinBERT** for sentiment.

If either the price download or the news/model download fails — for
example because you're on a network with outbound access restrictions, or
temporarily offline — **both scripts automatically and transparently fall
back** to clearly-labeled synthetic data (a realistic simulated price
series, and template-generated headlines scored by a small local model)
so the rest of the pipeline keeps working end-to-end rather than crashing.
Every output file and the dashboard itself show which mode was used
(`data_source` / `headline_source` / `model_source` columns and an on-screen
banner). **On a machine with normal internet access, you'll get real AAPL
price data and real FinBERT-scored news automatically — no code changes
needed.**

One consequence worth knowing: the synthetic price series and synthetic
headline series are generated independently of each other (there's no
built-in causal link between "simulated good news" and "simulated price
goes up"), so if you're seeing the fallback data, the what-if sentiment
slider will show only a small, mostly-noise-driven price sensitivity. With
real historical prices and real correlated news sentiment, this
relationship will be more meaningful.

## Project structure

```
neuroticker/
├── data_collection.py
├── features.py
├── sentiment.py
├── models.py
├── explain.py
├── app.py
├── requirements.txt
├── README.md
└── data/                      # created at runtime
    ├── raw_prices.csv
    ├── features.csv
    ├── sentiment.csv
    ├── sentiment_headlines.csv
    ├── forecast.json
    ├── xgb_model.json
    ├── shap_explanation.json
    ├── features_complete.csv
    └── train_snapshot.csv
```

## Disclaimer

This is a research / educational project. Nothing here is investment
advice. Stock price prediction is inherently noisy and uncertain — treat
the ensemble forecast and confidence interval as illustrative, not
reliable trading signals.
