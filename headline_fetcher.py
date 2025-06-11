 import os, json, requests, datetime as dt, pandas as pd, time

TODAY = dt.date.today().isoformat()

def grab_finnhub():
    key = os.getenv("FINNHUB_KEY", "").strip()
    if not key:
        print("Finnhub key missing"); return []
    url = f"https://finnhub.io/api/v1/news?category=general&token={key}"
    data = requests.get(url, timeout=15).json()
    if isinstance(data, dict) and "error" in data:
        print("Finnhub error:", data["error"]); return []
    return [(it["related"].split(",")[0], it["headline"]) for it in data]

def grab_alpha():
    key = os.getenv("AV_KEY", "").strip()
    if not key:
        return []
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={key}"
    data = requests.get(url, timeout=15).json().get("feed", [])
    return [(f["ticker"], f["title"]) for f in data]

rows = []
for fn in (grab_finnhub, grab_alpha):
    rows.extend({"sym": s, "hl": h} for s, h in fn())

df = pd.DataFrame(rows).drop_duplicates()
if df.empty:
    raise SystemExit("No headlines pulled — check API keys or quotas.")

HF_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_HEADERS = {"Accept": "application/json"}

def score(headline):
    payload = {"inputs": headline}
    try:
        r = requests.post(HF_URL, headers=HF_HEADERS,
                          data=json.dumps(payload), timeout=20)
        # HF might reply with 503 (model loading) or non-JSON; guard for that
        if r.status_code == 503:
            time.sleep(1); return 0            # neutral fallback
        result = r.json()
        if not isinstance(result, list):
            return 0
        label = result[0]["label"].lower()
        return {"positive": 1, "negative": -1}.get(label, 0)
    except (requests.exceptions.RequestException,
            json.JSONDecodeError, KeyError, IndexError):
        return 0   # neutral on any error

df["score"] = df["hl"].apply(score)
watch = (df.groupby("sym")["score"].mean()
           .reset_index().query("score > 0"))

outfile = f"longs_{TODAY}.csv"
watch.to_csv(outfile, index=False)
print(f"✅  Created {outfile} with {len(watch)} tickers.")
