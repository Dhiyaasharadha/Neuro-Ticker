"""
data_collection.py
-------------------
Downloads historical daily OHLCV price data for a stock ticker using yfinance
and saves it to data/raw_prices.csv.

Usage:
    python data_collection.py --ticker AAPL --start 2015-01-01

If yfinance/Yahoo Finance is unreachable (e.g. no internet access, or the
network blocks finance.yahoo.com), this script falls back to generating a
realistic synthetic price series so the rest of the pipeline can still be
built, tested, and demoed offline. The fallback is clearly logged and never
silently used.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd


def download_real_prices_alphavantage(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Preferred real-data source: Alpha Vantage's official TIME_SERIES_DAILY
    endpoint. Unlike yfinance (which scrapes Yahoo's undocumented internal
    API) or Stooq's CSV export (an unofficial convenience endpoint), this
    is a real, documented, key-authenticated API meant for programmatic
    access -- so it doesn't have the cloud-IP bot-detection problems that
    make scraping-based sources flaky on hosting platforms. Requires a
    free API key (https://www.alphavantage.co/support/#api-key) set as the
    ALPHAVANTAGE_API_KEY environment variable. If that env var isn't set,
    this source is skipped (not treated as an error).
    """
    import requests

    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY environment variable not set -- skipping this source")

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": "full",  # up to ~20 years of daily data in one call
        "apikey": api_key,
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()

    if "Time Series (Daily)" not in payload:
        note = payload.get("Note") or payload.get("Information") or payload.get("Error Message") or str(payload)[:200]
        raise RuntimeError(f"Alpha Vantage did not return time series data: {note}")

    series = payload["Time Series (Daily)"]
    rows = []
    for date_str, values in series.items():
        rows.append({
            "date": date_str,
            "open": float(values["1. open"]),
            "high": float(values["2. high"]),
            "low": float(values["3. low"]),
            "close": float(values["4. close"]),
            "adj_close": float(values["4. close"]),
            "volume": int(values["5. volume"]),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)

    if df.empty:
        raise RuntimeError(f"Alpha Vantage returned data, but none in the requested date range {start}..{end}")
    return df


def download_real_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download real historical prices via yfinance."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    # yfinance sometimes returns MultiIndex columns when a single ticker is
    # passed as a list, or in newer versions. Flatten if needed.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    df.rename(columns={"Date": "date"}, inplace=True)
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df


def download_real_prices_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fallback real-data source: Stooq's plain CSV export endpoint.

    Yahoo Finance's undocumented API (what yfinance scrapes) frequently
    rate-limits or outright blocks requests coming from cloud/datacenter IP
    ranges (Render, AWS, Railway, Heroku, ...) even though those servers
    have completely normal internet access -- it's Yahoo's bot detection,
    not a connectivity problem. Stooq's CSV export endpoint is a plain,
    unauthenticated HTTP download with no such bot-blocking historically,
    so it's a good second real-data attempt before giving up and using
    synthetic data.
    """
    import requests
    import io as _io

    start_compact = pd.to_datetime(start).strftime("%Y%m%d")
    end_compact = pd.to_datetime(end).strftime("%Y%m%d")
    symbol = f"{ticker.lower()}.us"

    url = (
        f"https://stooq.com/q/d/l/?s={symbol}&d1={start_compact}&d2={end_compact}&i=d"
    )
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    text = resp.text.strip()
    if not text or text.lower().startswith("no data") or "<html" in text.lower():
        raise RuntimeError(f"Stooq returned no usable data for {ticker} ({text[:80]!r})")

    df = pd.read_csv(_io.StringIO(text))
    if df.empty or "Close" not in df.columns:
        raise RuntimeError(f"Stooq CSV for {ticker} was empty or malformed")

    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"date": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df["adj_close"] = df["close"]
    return df


def generate_synthetic_prices(ticker: str, start: str, end: str, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic daily OHLCV series using a geometric
    Brownian motion model with mild drift, volatility clustering, and
    weekday-only trading days. Used only as an offline fallback when the
    real data source cannot be reached.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)  # business days only
    n = len(dates)

    # GBM-ish daily log returns with a little volatility clustering (GARCH-lite)
    mu = 0.0004
    base_vol = 0.017
    vol = np.zeros(n)
    vol[0] = base_vol
    shocks = rng.standard_normal(n)
    for i in range(1, n):
        vol[i] = 0.94 * vol[i - 1] + 0.06 * base_vol * (1 + abs(shocks[i - 1]))
    log_returns = mu + vol * shocks

    start_price = 27.0  # roughly AAPL's split-adjusted price in early 2015
    close = start_price * np.exp(np.cumsum(log_returns))

    daily_range = close * (0.006 + 0.004 * rng.random(n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) + daily_range * rng.random(n)
    low = np.minimum(open_, close) - daily_range * rng.random(n)
    volume = rng.integers(40_000_000, 160_000_000, n)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        }
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Download historical stock prices.")
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--out", default="data/raw_prices.csv", help="Output CSV path")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        default=True,
        help="Fall back to synthetic data if the real download fails (default: on)",
    )
    args = parser.parse_args()

    end = args.end or datetime.today().strftime("%Y-%m-%d")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[data_collection] Ticker={args.ticker}  Start={args.start}  End={end}")

    df = None
    source = None
    try:
        print("[data_collection] Attempting real download via Alpha Vantage (official API) ...")
        df = download_real_prices_alphavantage(args.ticker, args.start, end)
        print(f"[data_collection] SUCCESS: downloaded {len(df)} real rows from Alpha Vantage.")
        source = "alphavantage (real)"
    except Exception as e0:
        print(f"[data_collection] Alpha Vantage failed/skipped: {e0}")
        try:
            print("[data_collection] Attempting real download via yfinance ...")
            df = download_real_prices(args.ticker, args.start, end)
            print(f"[data_collection] SUCCESS: downloaded {len(df)} real rows from Yahoo Finance (yfinance).")
            source = "yfinance (real)"
        except Exception as e:
            print(f"[data_collection] yfinance download failed: {e}")
            try:
                print("[data_collection] Attempting real download via Stooq (fallback data source) ...")
                df = download_real_prices_stooq(args.ticker, args.start, end)
                print(f"[data_collection] SUCCESS: downloaded {len(df)} real rows from Stooq.")
                source = "stooq (real)"
            except Exception as e2:
                print(f"[data_collection] Stooq download also failed: {e2}")
                if not args.allow_synthetic:
                    sys.exit(1)
                print("[data_collection] All real sources failed. Falling back to SYNTHETIC data for offline testing.")
                df = generate_synthetic_prices(args.ticker, args.start, end)
                source = "synthetic (offline fallback)"

    df["ticker"] = args.ticker
    df["data_source"] = source
    df.to_csv(args.out, index=False)

    print(f"[data_collection] Saved {len(df)} rows -> {args.out}")
    print(f"[data_collection] Data source: {source}")
    print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
