import os, json, requests, datetime as dt, pandas as pd

TODAY = dt.date.today().isoformat()

def grab_finnhub():
    raw = (os.getenv("FINNHUB_KEY") or "").strip()       # <- strip hidden \n
    if not raw:
        print("Finnhub key missing")
        return []
    url = f"https://finnhub.io/api/v1/news?category=general&token={raw}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        # Debug: show first 60 chars of Finnhub reply
        print("Finnhub reply:", str(data)[:60], "…")
        # If Finnhub returns {"error": "..."} treat as empty
        if isinstance(data, dict) and "error" in data:
            return []
        return [(item["related"].split(",")[0], item["headline"])
                for item in data]
    except Exception as e:
        print("Finnhub fetch failed:", e)
        return []

def grab_alpha():
    raw = (os.getenv("AV_KEY") or "").strip()
    if not raw:
        return []
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={raw}"
    try:
        feed = requests.get(url, timeout=10).json().get("feed", [])
        return [(f["ticker"], f["title"]) for f in feed]
    except Exception:
        return []

rows = []
for fn in (grab_finnhub, grab_alpha):
    rows.extend({"sym": s, "hl": h} for s, h in fn())

df = pd.DataFrame(rows).drop_duplicates()
if df.empty:
    raise SystemExit("No headlines pulled — check API keys and quotas.")

def score(headline):
    payload = {"inputs": headline}
    r = requests.post(
        "https://api-inference.huggingface.co/models/ProsusAI/finbert",
        headers={"Accept": "application/json"},
        data=json.dumps(payload), timeout=20).json()
    label = r[0]["label"].lower() if isinstance(r, list) else "neutral"
    return {"positive": 1, "negative": -1}.get(label, 0)

df["score"] = df["hl"].apply(score)
watch = (df.groupby("sym")["score"].mean()
           .reset_index().query("score > 0"))

outfile = f"longs_{TODAY}.csv"
watch.to_csv(outfile, index=False)
print(f"✅  Created {outfile} with {len(watch)} tickers.")