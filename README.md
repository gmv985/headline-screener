# Headline Screener

Tiny, free, cloud-hosted tool that:
1. Pulls today’s equity headlines from free APIs (Finnhub, Alpha Vantage).
2. Runs each headline through the open-source **FinBERT** model.
3. Saves a CSV of tickers whose average sentiment > 0.

The GitHub Actions workflow (`.github/workflows/daily.yml`)
runs every weekday at 06:05 US Eastern and uploads the CSV as an artifact.

## Download price data to CSV

Use the included helper to grab historical prices (default: 30 days of BTC-USD)
and save them locally:

```bash
python download_prices.py                 # writes prices.csv for BTC-USD
python download_prices.py ETH-USD --period 90d --interval 1h --output eth.csv
```
