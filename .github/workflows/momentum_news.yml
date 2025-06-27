#!/usr/bin/env python3
"""
Pulls S&P 500 tickers, checks today’s intraday return with yfinance,
computes 24 h FinBERT sentiment from Finnhub headlines and writes two lists:
  longs_<YYYY-MM-DD>.csv  – best positive momentum & sentiment
  shorts_<YYYY-MM-DD>.csv – worst negative momentum & sentiment
Needs a FINNHUB_KEY env var (free key works, quota is low).
"""
import os, sys, datetime as dt, requests, pandas as pd, yfinance as yf
from transformers import pipeline

API = "https://finnhub.io/api/v1/news?category=general&symbol={tkr}&token={key}"

# ---------- helpers ---------------------------------------------------------
def sp500():
    """Return current S&P 500 symbols from Wikipedia."""
    return pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"]

def intraday_ret(tkr):
    end = dt.datetime.utcnow(); start = end - dt.timedelta(days=3)
    px = yf.download(tkr, start=start, end=end, interval="30m",
                     progress=False)["Adj Close"]
    return None if px.empty else px.iloc[-1]/px.iloc[0]-1

def headlines(tkr, key):
    r = requests.get(API.format(tkr=tkr, key=key), timeout=20).json()
    return [item["headline"] for item in r][:20]

def finbert():
    print("Loading FinBERT …")
    return pipeline("sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    device="cpu")

def score(hls, model):
    if not hls: return 0
    m = {"positive": 1, "neutral": 0, "negative": -1}
    out = model(hls, truncation=True)
    return sum(m[o["label"].lower()] * o["score"] for o in out) / len(out)
# ---------------------------------------------------------------------------

def main():
    key = os.getenv("FINNHUB_KEY")
    if not key: sys.exit("FINNHUB_KEY not set")

    model = finbert()
    rows = []
    for t in sp500():
        r = intraday_ret(t);  h = headlines(t, key) if r is not None else None
        if h is None: continue
        rows.append({"ticker": t, "ret": r, "sent": score(h, model)})

    df = pd.DataFrame(rows)
    today = dt.date.today().isoformat()
    df.query("ret>0 and sent>0").sort_values(["ret","sent"], ascending=False)\
      .head(10).to_csv(f"longs_{today}.csv", index=False)
    df.query("ret<0 and sent<0").sort_values(["ret","sent"])\
      .head(10).to_csv(f"shorts_{today}.csv", index=False)
    print("Done.")

if __name__ == "__main__":
    main()
