import os, json, requests, datetime as dt, pandas as pd, time, sys

TODAY = dt.date.today().isoformat()

# ---------- headline feeders ----------
def grab_finnhub():
    k = os.getenv("FINNHUB_KEY", "").strip()
    if not k:
        return []
    url = f"https://finnhub.io/api/v1/news?category=general&token={k}"
    data = requests.get(url, timeout=15).json()
    return [
        (item.get("related", "").split(",")[0] or "UNKNOWN", item["headline"])
        for item in data if "headline" in item
    ]

def grab_alpha():
    k = os.getenv("AV_KEY", "").strip()
    if not k:
        return []
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={k}"
    data = requests.get(url, timeout=15).json().get("feed", [])
    return [
        (f.get("ticker", "UNKNOWN"), f["title"])
        for f in data if "title" in f
    ]

rows = []
for fn in (grab_finnhub, grab_alpha):
    rows.extend({"sym": s, "hl": h} for s, h in fn())

df = pd.DataFrame(rows).drop_duplicates()
if df.empty:
    for name in ("longs", "shorts"):
        pd.DataFrame(columns=["sym","score"]).to_csv(f"{name}_{TODAY}.csv", index=False)
    print("No headlines pulled – empty CSVs written.")
    sys.exit(0)

# ---------- FinBERT sentiment ----------
HF_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HDR = {"Accept": "application/json"}

def score(text):
    try:
        r = requests.post(HF_URL, headers=HDR,
                          data=json.dumps({"inputs": text}), timeout=20)
        if r.status_code == 503:
            time.sleep(1); return 0
        label = r.json()[0]["label"].lower()
        return {"positive": 1, "negative": -1}.get(label, 0)
    except Exception:
        return 0

df["score"] = df["hl"].apply(score)
avg = df.groupby("sym")["score"].mean().reset_index()

longs  = avg.query("score > 0")
shorts = avg.query("score < 0")

longs.to_csv (f"longs_{TODAY}.csv",  index=False)
shorts.to_csv(f"shorts_{TODAY}.csv", index=False)

print(f"✅  Longs:  {len(longs)}   Shorts: {len(shorts)}")