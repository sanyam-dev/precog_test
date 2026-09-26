## Task 5: mapping anonymized assets to S&P 500 tickers

The anonymized closes are real dividend-adjusted closes times a per-asset constant, so daily log returns are unchanged. `task5.ipynb` matches each asset to an S&P 500 stock (or one of a few large ADRs) by return correlation, then checks that anonymized / real price is constant.

- `yahoodownloader.py`: FinRL's `YahooDownloader`, vendored and fixed for current yfinance
- `mapping.py`: loading, return correlation, one-to-one assignment (Hungarian), price-scale check
- `artifacts/asset_mapping.csv`: asset, ticker, name, sector, corr, runner-up, scale
- `artifacts/universe.csv`, `artifacts/yahoo_prices.csv.gz`: cached ticker list and Yahoo prices. The notebook loads them if both exist, else downloads and saves them. Prices are git-ignored; delete both to refetch

Run the notebook from inside `task5/`.
