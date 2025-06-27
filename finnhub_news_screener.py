#!/usr/bin/env python3
"""
1. Pull S&P-500 constituent list (free from Wikipedia)
2. Compute 1-day momentum (yesterday close minus close-2-days-ago)
3. Rank top 30 & bottom 30 symbols
4. Fetch last-24h headlines from Finnhub for just those 60 tickers
5. Score each headline with FinBERT → aggregate to NewsScore 0-10
6. Produce finnhub_longlist.csv  (bullish + bearish together)
"""

import os, datetime as dt, requests, pandas as pd, numpy as np
import yfinance as yf
from transformers import pipeline

FINNHUB = os.getenv("FINNHUB_KEY")
HEADERS  = {"X-Finnhub-Token": FINNHUB}

TODAY   = dt.date.today().isoformat()

# ---------- 1) S&P-500 tickers ----------
sp500 = (pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
         ["Symbol"].tolist())

# ---------- 2) 1-day momentum ----------
data = yf.download(sp500, period="3d", interval="1d", group_by="ticker", progress=False)
rows = []
for sym in sp500:
    try:
        closes = data[sym]["Close"].dropna()
        if len(closes) >= 3:
            diff_pct = (closes[-1] - closes[-2]) / closes[-2] * 100
            rows.append({"symbol": sym, "momentum_pct": diff_pct})
    except KeyError:
        continue
mom = pd.DataFrame(rows)

top30  = mom.sort_values("momentum_pct", ascending=False).head(30)
bot30  = mom.sort_values("momentum_pct").head(30)
cands  = pd.concat([top30, bot30]).reset_index(drop=True)

# ---------- 3) FinBERT sentiment ----------
finbert = pipeline("sentiment-analysis",
                   model="ProsusAI/finbert",
                   top_k=None)

def fetch_headlines(ticker):
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={TODAY}&to={TODAY}"
    try:
        js = requests.get(url, headers=HEADERS, timeout=10).json()
        return [n["headline"] for n in js]
    except Exception:
        return []

def score_news(titles):
    pos = neg = 0
    for t in titles:
        label = finbert(t)[0]["label"].lower()
        if label == "positive":  pos += 1
        elif label == "negative": neg += 1
    tot = pos + neg
    if tot == 0:
        return 5.0          # neutral by definition
    # 0-10 scale
    return round(abs(pos - neg) / tot * 10, 1)

news_scores = []
for sym in cands["symbol"]:
    ns = score_news(fetch_headlines(sym))
    news_scores.append(ns)

cands["NewsScore"] = news_scores

# ---------- 4) export ----------
cands.to_csv("finnhub_longlist.csv", index=False)
print(cands.head(10))
