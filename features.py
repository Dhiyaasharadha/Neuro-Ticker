"""
features.py
------------
Engineers technical indicators from data/raw_prices.csv and saves the result
to data/features.csv.

Features created:
    - ma_7   : 7-day simple moving average of close price
    - ma_30  : 30-day simple moving average of close price
    - vol_7  : 7-day rolling volatility (std dev of daily returns)
    - rsi_14 : 14-day Relative Strength Index
    - return_1d : 1-day pct change (used as a helper/label feature)

Usage:
    python features.py --in data/raw_prices.csv --out data/features.csv
"""

import argparse
import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Standard Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral RSI where undefined (start of series)
    return rsi


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    close_col = "adj_close" if "adj_close" in df.columns else "close"

    df["return_1d"] = df[close_col].pct_change()
    df["ma_7"] = df[close_col].rolling(window=7, min_periods=1).mean()
    df["ma_30"] = df[close_col].rolling(window=30, min_periods=1).mean()
    df["vol_7"] = df["return_1d"].rolling(window=7, min_periods=2).std()
    df["rsi_14"] = compute_rsi(df[close_col], period=14)

    # next-day close is the prediction target used later by models.py
    df["target_next_close"] = df[close_col].shift(-1)

    df["vol_7"] = df["vol_7"].bfill().fillna(0)
    df["return_1d"] = df["return_1d"].fillna(0)

    keep_cols = [
        "date", "ticker", close_col, "open", "high", "low", "volume",
        "return_1d", "ma_7", "ma_30", "vol_7", "rsi_14", "target_next_close",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].rename(columns={close_col: "close"})


def main():
    parser = argparse.ArgumentParser(description="Engineer technical indicator features.")
    parser.add_argument("--in", dest="infile", default="data/raw_prices.csv")
    parser.add_argument("--out", dest="outfile", default="data/features.csv")
    args = parser.parse_args()

    print(f"[features] Loading {args.infile}")
    df = pd.read_csv(args.infile)

    feat_df = build_features(df)
    feat_df.to_csv(args.outfile, index=False)

    print(f"[features] Saved {len(feat_df)} rows x {len(feat_df.columns)} cols -> {args.outfile}")
    print(feat_df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
