import os, json, requests, datetime as dt, pandas as pd, time

TODAY = dt.date.today().isoformat()

def grab_finnhub():
    key = os.getenv("FINNHUB_KEY", "").strip()
    if not key:
        print("Finnhub key missing"); return []
    url = f"https://finnhub.io/api/v1/news?category=general&token={key}"
    data = requests.get(url, timeout=15).json()
    if isinstance(data, dict) and "error" in data:
        print("Finnhub error:", data['error']); return []
    out = []
    for it in data:
        sym = it.get("related", "").split(",")[0] or "UNKNOWN"
        out.append((sym, it["headline"]))
    return out

def grab_alpha():
    key = os.getenv("AV_KEY", "").strip()
    if not key:
        return []
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={key}"
    data = requests.get(url, timeout=15).json().get("feed", [])
    return [(f.get("ticker","UNKNOWN"), f["title"]) for f in data]

rows = []
for fn in (grab_finnhub, grab_alpha):
    rows.extend({"sym": s, "hl": h} for s, h in fn())

df = pd.DataFrame(rows).drop_duplicates()

# If still empty, just create an empty CSV instead of exiting
if df.empty:
    pd.DataFrame(columns=["sym","score"]).to_csv(f"longs_{TODAY}.csv", index=False)
    print("✅  No tickers today, empty CSV created.")
    raise SystemExit

# ---------- FinBERT sentiment ----------
HF_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_HEADERS = {"Accept": "application/json"}

def score(headline):
    try:
        r = requests.post(HF_URL, headers=HF_HEADERS,
                          data=json.dumps({"inputs": headline}), timeout=20)
        if r.status_code == 503:
            time.sleep(1); return 0
        result = r.json()
        if not isinstance(result, list):
            return 0
        label = result[0]["label"].lower()
        return {"positive": 1, "negative": -1}.get(label, 0)
    except Exception:
        return 0

df["score"] = df["hl"].apply(score)
watch = (df.groupby("sym")["score"].mean()
           .reset_index().query("score > 0"))

outfile = f"longs_{TODAY}.csv"
watch.to_csv(outfile, index=False)
print(f"✅  Created {outfile} with {len(watch)} tickers.")