#!/usr/bin/env python
"""
Free Momentum + News-sentiment screener
• picks the 30 strongest-momentum S&P-500 stocks (15-day look-back)
• scores their headlines with FinBERT
• writes daily_long_list.csv
Env-var required: FINNHUB_KEY
"""

import os, time, datetime as dt, requests, pandas as pd
from pathlib import Path

# ---------- 1  check API key ----------
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
if not FINNHUB_KEY:
    raise RuntimeError("FINNHUB_KEY environment variable not set")

# ---------- 2  get S&P-500 constituents ----------
sp500_url = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents_symbols.txt"
)
TICKERS = requests.get(sp500_url, timeout=20).text.strip().splitlines()

# ---------- 3  simple 15-day momentum ----------
def fetch_close(ticker: str, days: int = 15):
    end = int(time.time())
    start = end - days * 86_400
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={start}&period2={end}&interval=1d"
    )
    j = requests.get(url, timeout=20).json()
    closes = (
        j["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        if j.get("chart", {}).get("result")
        else []
    )
    return closes if None not in closes else []

momentum = []
for t in TICKERS:
    try:
        px = fetch_close(t)
        if len(px) >= 2:
            momentum.append((t, px[-1] / px[0] - 1))
    except Exception:
        pass

momentum_df = pd.DataFrame(momentum, columns=["ticker", "momentum"]).sort_values(
    "momentum", ascending=False
)
top30 = momentum_df.head(30)["ticker"].tolist()

# ---------- 4  Finhub headlines ----------
def finnhub_headlines(ticker: str, hours_back: int = 24):
    now = dt.datetime.utcnow()
    start = (now - dt.timedelta(hours=hours_back)).strftime("%Y-%m-%d")
    url = (
        f"https://finnhub.io/api/v1/news?symbol={ticker}&from={start}"
        f"&token={FINNHUB_KEY}"
    )
    r = requests.get(url, timeout=20)
    return [h["headline"] for h in r.json()] if r.status_code == 200 else []

# ---------- 5  FinBERT sentiment ----------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
LABELS = ["negative", "neutral", "positive"]

def score_headlines(headlines):
    if not headlines:
        return 0.0
    s = 0
    for txt in headlines:
        tok = tokenizer(txt, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**tok).logits.squeeze().cpu()
        prob = torch.softmax(logits, dim=0)
        s += float(prob[2] - prob[0])  # Positive – Negative
    avg = s / len(headlines)
    return round(max(min((avg + 1) / 2 * 10, 10), 0), 2)  # scale to 0-10

# ---------- 6  assemble results ----------
rows = []
for t in top30:
    rows.append(
        {
            "ticker": t,
            "momentum": momentum_df.loc[momentum_df.ticker == t, "momentum"].values[0],
            "news_score": score_headlines(finnhub_headlines(t)),
        }
    )

out = pd.DataFrame(rows).sort_values("news_score", ascending=False)
outfile = Path("daily_long_list.csv")
out.to_csv(outfile, index=False)
print(f"Saved {len(out)} rows → {outfile.resolve()}")
