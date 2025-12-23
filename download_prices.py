#!/usr/bin/env python3
"""
Download historical price data to CSV using yfinance.

Defaults fetch 30 days of daily candles for BTC-USD, but the ticker,
period, interval, and output file are configurable via CLI flags.
"""
import argparse
import sys

import pandas as pd
import yfinance as yf


def fetch_prices(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Return price history for the given ticker.

    Parameters
    ----------
    ticker: str
        Symbol understood by Yahoo Finance (e.g., ``BTC-USD``).
    period: str
        Time span such as ``30d`` or ``1y``.
    interval: str
        Bar size such as ``1d`` or ``1h``.
    """
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker} ({period=}, {interval=})")
    return data.reset_index()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download price data to CSV")
    parser.add_argument("ticker", nargs="?", default="BTC-USD",
                        help="Ticker symbol (default: BTC-USD)")
    parser.add_argument("--period", default="30d",
                        help="Lookback period, e.g., 30d, 1y (default: 30d)")
    parser.add_argument("--interval", default="1d",
                        help="Bar interval, e.g., 1d, 1h (default: 1d)")
    parser.add_argument("--output", default="prices.csv",
                        help="Output CSV filename (default: prices.csv)")
    args = parser.parse_args(argv)

    try:
        df = fetch_prices(args.ticker, args.period, args.interval)
    except Exception as exc:  # noqa: BLE001  (surface clean message to user)
        sys.exit(f"Error fetching data: {exc}")

    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
