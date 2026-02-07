#!/usr/bin/env python3
"""
Screen a UCITS ETF universe for yield >= 5%, score candidates, and export CSV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf

UNIVERSE_CSV = Path("ucits_etfs.csv")
OUTPUT_CSV = Path("ucits_etf_screen.csv")


@dataclass(frozen=True)
class EtfRow:
    ticker: str
    name: str


DEFAULT_UNIVERSE: list[EtfRow] = [
    EtfRow("IUKD.L", "iShares UK Dividend UCITS ETF"),
    EtfRow("IDVY.L", "iShares Euro Dividend UCITS ETF"),
    EtfRow("VHYL.L", "Vanguard FTSE All-World High Dividend Yield UCITS ETF"),
    EtfRow("ISPA.L", "iShares Asia Pacific Dividend UCITS ETF"),
    EtfRow("IQDY.L", "iShares U.S. Dividend IQ UCITS ETF"),
    EtfRow("HDLV.L", "Invesco FTSE EM High Dividend Low Volatility UCITS ETF"),
    EtfRow("TDIV.L", "VanEck Morningstar Developed Markets Dividend Leaders UCITS ETF"),
    EtfRow("UDVD.L", "SPDR S&P U.S. Dividend Aristocrats UCITS ETF"),
]


def load_universe() -> list[EtfRow]:
    if not UNIVERSE_CSV.exists():
        return DEFAULT_UNIVERSE
    data = pd.read_csv(UNIVERSE_CSV)
    if {"ticker", "name"}.issubset(data.columns):
        return [EtfRow(row.ticker, row.name) for row in data.itertuples(index=False)]
    raise ValueError("ucits_etfs.csv must contain 'ticker' and 'name' columns")


def get_info_value(info: dict, *keys: str) -> Optional[float]:
    for key in keys:
        value = info.get(key)
        if value is not None:
            return value
    return None


def fetch_metrics(ticker: str) -> dict:
    info = yf.Ticker(ticker).get_info()
    dividend_yield = get_info_value(info, "dividendYield", "trailingAnnualDividendYield")
    avg_volume = get_info_value(
        info,
        "averageVolume",
        "averageDailyVolume10Day",
        "averageVolume10days",
    )
    expense_ratio = get_info_value(info, "annualReportExpenseRatio")
    return {
        "dividend_yield": dividend_yield,
        "avg_volume": avg_volume,
        "expense_ratio": expense_ratio,
        "currency": info.get("currency"),
    }


def score_yield(yield_pct: float) -> float:
    return round(min(max(yield_pct, 0), 10), 2)


def score_liquidity(avg_volume: Optional[float]) -> float:
    if not avg_volume or avg_volume <= 0:
        return 0.0
    log_vol = math.log10(avg_volume)
    scaled = (log_vol - 3) / 3 * 10
    return round(min(max(scaled, 0), 10), 2)


def score_total(yield_score: float, liquidity_score: float) -> float:
    return round(yield_score * 0.7 + liquidity_score * 0.3, 2)


def build_rows(universe: Iterable[EtfRow]) -> list[dict]:
    rows: list[dict] = []
    for etf in universe:
        metrics = fetch_metrics(etf.ticker)
        yield_raw = metrics["dividend_yield"]
        yield_pct = None if yield_raw is None else round(yield_raw * 100, 2)
        liquidity_score = score_liquidity(metrics["avg_volume"])
        if yield_pct is None:
            yield_score = None
            total_score = None
        else:
            yield_score = score_yield(yield_pct)
            total_score = score_total(yield_score, liquidity_score)
        rows.append(
            {
                "ticker": etf.ticker,
                "name": etf.name,
                "currency": metrics["currency"],
                "dividend_yield_pct": yield_pct,
                "avg_volume": metrics["avg_volume"],
                "expense_ratio": metrics["expense_ratio"],
                "yield_score": yield_score,
                "liquidity_score": liquidity_score,
                "total_score": total_score,
            }
        )
    return rows


def main() -> None:
    universe = load_universe()
    rows = build_rows(universe)
    df = pd.DataFrame(rows)
    screened = df[df["dividend_yield_pct"].fillna(0) >= 5].copy()
    screened = screened.sort_values("total_score", ascending=False)
    screened.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(screened)} ETFs to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
