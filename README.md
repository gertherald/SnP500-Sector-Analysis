# Do Sectors Describe How Stocks Move?

How well do official GICS sectors describe how S&P 500 stocks actually move, which stocks behave unlike their sector, and how stable are those groupings? Stocks are clustered by their daily returns and the clusters are compared with their sectors.

## Data
- **Prices:** about two years of adjusted daily closes (2024-11-18 to 2026-11-18) from Yahoo Finance via `yfinance`
- **Sectors:** current S&P 500 constituents with GICS sector, scraped from Wikipedia

## Setup
Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
Run the notebooks in order from the `notebooks/` folder:

```bash
jupyter notebook notebooks/01_data_pipeline.ipynb
```

Or run only the data pipeline from the project root:

```bash
python -m src.data
```

The first run downloads data (about a minute) and caches it in `data/raw/` as Parquet. Later runs load the cache. To download again, call `data.run_pipeline(force_refresh=True)`.

## Outputs
| File | Contents |
|---|---|
| `data/raw/sp500_sectors.parquet` | ticker, company, sector, sub_industry |
| `data/raw/prices.parquet` | date, ticker, close, volume (long format) |
| `data/processed/returns_long.parquet` | one row per stock-day with daily return |
| `data/processed/returns_matrix.parquet` | stocks x days matrix of daily returns |
| `data/processed/sectors.parquet` | sectors of the stocks kept after cleaning |
| `data/processed/clusters.parquet` | ticker, sector, KMeans cluster and PCA scores per stock |

## Layout
```
src/config.py      fixed dates, thresholds, seed, paths
src/data.py        data pipeline (Polars)
src/baseline.py    correlation baseline: heatmap, within vs cross-sector correlation
src/clustering.py  standardising, PCA with shuffle noise check, KMeans, silhouette scores
notebooks/         analysis notebooks, run in order
docs/              proposal, process book
data/              cached and processed data (not in git)
```

## Note on reproducibility
Settings live in `src/config.py`. Wikipedia reflects the index as of the download date, so re-downloading later can change the stock list. Keep the cached `data/raw/` files to reproduce results exactly.
