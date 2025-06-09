import os, requests, datetime as dt, pandas as pd
from transformers import pipeline

TODAY = dt.date.today().isoformat()
finbert = pipeline("sentiment-analysis",
                   model="ProsusAI/finbert",
                   truncation=True, do_lower_case=True)

def grab_finnhub():
    key = os.getenv("FINNHUB_KEY")
    if not key:
        return []
    url = f"https://finnhub.io/api/v1/news?category=general&token={key}"
    items = requests.get(url, timeout=10).json()
    return [(x["related"].split(",")[0], x["headline"]) for x in items]

def grab_alpha():
    key = os.getenv("AV_KEY")
    if not key:
        return []
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={key}"
    feed = requests.get(url, timeout=10).json().get("feed", [])
    return [(a["ticker"], a["title"]) for a in feed]

FETCHERS = [grab_finnhub, grab_alpha]

rows = []
for fn in FETCHERS:
    rows.extend({"sym": s, "hl": h} for s, h in fn())

df = pd.DataFrame(rows).drop_duplicates()
if df.empty:
    raise SystemExit("No headlines pulled – check API keys / quotas.")

def score(text):
    sentiment = finbert(text)[0]["label"].lower()
    return {"positive": 1, "negative": -1}.get(sentiment, 0)

df["score"] = df["hl"].apply(score)
watch = (df.groupby("sym")["score"].mean()
           .reset_index()
           .query("score > 0"))

outfile = f"longs_{TODAY}.csv"
watch.to_csv(outfile, index=False)
print("✅  Created", outfile, "with", len(watch), "tickers.")
