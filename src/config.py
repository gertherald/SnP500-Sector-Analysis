"""Fixed settings for the whole project, so every run uses the same data and seeds."""

from pathlib import Path

# --- Date range -------------------------------------------------------------
# Two years of daily prices, ending on a fixed date so results are reproducible.
# Both dates are inclusive. (yfinance treats `end` as exclusive, data.py adds one day.)
START_DATE = "2024-11-18"
END_DATE = "2026-11-18"

# --- Cleaning rules ---------------------------------------------------------
# Drop a stock if more than this share of trading days has no closing price
# (for example companies that listed or joined the market during the window).
MISSING_THRESHOLD = 0.05

# Companies listed twice (two share classes of the same company move almost identically).
# Keep the Class A ticker (GOOGL, FOXA, NWSA) and drop these duplicates.
DUPLICATE_SHARE_CLASSES = ["GOOG", "FOX", "NWS"]

# A daily return above this (in absolute value) is flagged for manual review.
EXTREME_RETURN = 0.50

# --- Reproducibility --------------------------------------------------------
RANDOM_SEED = 42

# --- Clustering -------------------------------------------------------------
MAX_COMPONENTS = 60  # PCA components computed before deciding how many to keep
N_SHUFFLES = 20  # shuffled copies of the data used as the noise benchmark
NOISE_PERCENTILE = 95  # a real component must beat this percentile of the shuffled ones
N_CLUSTERS = 11  # same number as GICS sectors, so clusters and sectors compare like for like
KMEANS_N_INIT = 10  # KMeans restarts from different starting centres, best one kept

# --- Data sources -----------------------------------------------------------
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia blocks the default Python User-Agent, so identify the client politely.
USER_AGENT = "SnP500-Sector-Analysis/0.1 (university data project; python-requests)"
DOWNLOAD_BATCH_SIZE = 100

# --- File paths -------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_SECTORS_PATH = RAW_DIR / "sp500_sectors.parquet"
RAW_PRICES_PATH = RAW_DIR / "prices.parquet"

RETURNS_LONG_PATH = PROCESSED_DIR / "returns_long.parquet"
RETURNS_MATRIX_PATH = PROCESSED_DIR / "returns_matrix.parquet"
SECTORS_PATH = PROCESSED_DIR / "sectors.parquet"
CLUSTERS_PATH = PROCESSED_DIR / "clusters.parquet"
