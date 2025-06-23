#!/usr/bin/env python3
"""
Pull FinBrain daily price-prediction universe,
blend in FinBrain news-sentiment,
add a 0-10 NewsScore column,
export to finbrain_longlist.csv
"""

import os, numpy as np, pandas as pd
from finbrain import Finbrain

fb = Finbrain(api_key=os.getenv("FINBRAIN_KEY"))

# ----- 1) Price-prediction universe (all symbols) -----
pred = fb.get_price_predictions(interval="daily")        # DataFrame
long_list = (pred
             .sort_values("prediction_diff_pct", ascending=False)
             .head(50)                                   # top 50 signals – change if desired
             .reset_index(drop=True))

# ----- 2) News sentiment for those symbols -----
sent = fb.get_news_sentiment(symbols=long_list["symbol"].tolist())
df = long_list.merge(sent, on="symbol", how="left")       # adds sentiment_score col

# ----- 3) Convert sentiment_score (-1…+1) -> 0…10 scale -----
def score(x):
    if pd.isna(x):
        return 5.0                # neutral if no news
    return np.clip((x + 1) * 5, 0, 10)

df["NewsScore"] = df["sentiment_score"].apply(score)

# keep only the useful columns
df = df[["symbol", "prediction_diff_pct", "NewsScore"]]

df.to_csv("finbrain_longlist.csv", index=False)
print(df.head(10))
