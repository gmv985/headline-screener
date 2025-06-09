import os, json, requests, datetime as dt, pandas as pd

TODAY = dt.date.today().isoformat()

# ---------- free headline fetchers ----------
def grab_finnhub():
    key = os.getenv("FINNHUB_KEY")
    if not key:
        return []
    url = f"https://finnhub.io/api/v1/news?category=general&token={key}"
    try:
        items = requests.get(url, timeout=10).json()
    except Exception:
        return []
    return [(x["related"].split(",")[0], x["headline"]) for x in items]

def grab_alpha():
    key = os.getenv("AV_KEY")
    if not key:
        return []
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={key}"
    try:
        feed = requests.get(url, timeout=10).json().get("feed", [])
    except Exception:
        return []
    return [(a["ticker"], a["title"]) for a in feed]

FETCHERS = [grab_finnhub, grab_alpha]

# ---------- FinBERT over Hugging-Face API ----------
HF_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
def score(headline):
    try:
        r = requests.post(HF_URL,
                          headers={"Accept": "application/json"},
                          data=json.dumps({"inputs": headline}),
                          timeout=20).json()
        label = r[0]['label'].lower()         # positive / negative / neutral
    except Exception:
        label = "neutral"
    return {"positive": 1, "negative": -1}.get(label, 0)
# ---------------------------------------------------

rows = []
for fn in FETCHERS:
    rows.extend({"sym": s, "hl": h} for s, h in fn())

df = pd.DataFrame(rows).drop_duplicates()
if df.empty:
    raise SystemExit("No headlines pulled – check API keys.")

df["score"] = df["hl"].apply(score)
watch = (df.groupby("sym")["score"].mean()
           .reset_index()
           .query("score > 0"))

outfile = f"longs_{TODAY}.csv"
watch.to_csv(outfile, index=False)
print("✅  Created", outfile, "with", len(watch), "tickers.")