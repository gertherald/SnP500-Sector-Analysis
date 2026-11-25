"""Data pipeline: S&P 500 sectors (Wikipedia) + daily prices (yfinance) -> daily returns.

Steps
1. fetch_sectors()           scrape the S&P 500 list with sectors (BeautifulSoup)
2. fetch_prices()            download daily prices (yfinance), cached to Parquet
3. build_returns_pipeline()  one lazy Polars query: join, drop gappy stocks, returns
4. pivot_returns()           long table -> matrix with one row per stock, one column per day
5. sanity-check helpers      missing_report(), null_report(), spot_check_return(), extreme_returns()
"""

from datetime import date, timedelta  # to work with the end date

import polars as pl  # all data processing
import requests  # to download the Wikipedia page
import yfinance as yf  # to download stock prices
from bs4 import BeautifulSoup  # to read the HTML of the Wikipedia page

from src import config  # project settings (dates, threshold, file paths)


# ---------------------------------------------------------------------------
# 1. Sector list (Wikipedia, scraped with BeautifulSoup)
# ---------------------------------------------------------------------------
def fetch_sectors(force_refresh: bool = False) -> pl.DataFrame:
    """Return one row per S&P 500 company: ticker, company, sector, sub_industry."""
    # If we already saved the list before, load it instead of scraping again
    if config.RAW_SECTORS_PATH.exists() and not force_refresh:
        return pl.read_parquet(config.RAW_SECTORS_PATH)

    # Download the page. Wikipedia blocks the default Python User-Agent, so we send our own
    headers = {"User-Agent": config.USER_AGENT}
    response = requests.get(config.WIKI_URL, headers=headers, timeout=30)
    response.raise_for_status()  # stop with an error if the download failed

    # Parse the HTML and find the table with id="constituents" (the S&P 500 list)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", id="constituents")

    rows = []  # one dictionary per company
    for tr in table.find_all("tr")[1:]:  # every table row, skipping the header row
        cells = tr.find_all("td")  # the cells of this row
        # .text is all the text inside a cell; .strip() removes spaces and line breaks at the ends
        rows.append(
            {
                "ticker": cells[0].text.strip(),  # column 1: Symbol
                "company": cells[1].text.strip(),  # column 2: Security
                "sector": cells[2].text.strip(),  # column 3: GICS Sector
                "sub_industry": cells[3].text.strip(),  # column 4: GICS Sub-Industry
            }
        )

    sectors = pl.DataFrame(rows)  # list of dictionaries -> Polars DataFrame

    # Yahoo writes share classes with "-" where Wikipedia uses "." (BRK.B -> BRK-B)
    sectors = sectors.with_columns(pl.col("ticker").str.replace(".", "-", literal=True))
    sectors = sectors.sort("ticker")  # alphabetical order

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)  # create data/raw/ if needed
    sectors.write_parquet(config.RAW_SECTORS_PATH)  # save so we don't scrape again
    return sectors


# ---------------------------------------------------------------------------
# 2. Prices (yfinance)
# ---------------------------------------------------------------------------
def fetch_prices(tickers: list[str], force_refresh: bool = False) -> pl.DataFrame:
    """Return daily prices in long format: one row per (date, ticker) with close and volume.

    Days without a price keep a row with close = null, so missing data stays visible.
    """
    # If we already downloaded the prices, load them instead of downloading again
    if config.RAW_PRICES_PATH.exists() and not force_refresh:
        return pl.read_parquet(config.RAW_PRICES_PATH)

    # yfinance's `end` date is not included, so ask for one day past END_DATE
    end = date.fromisoformat(config.END_DATE) + timedelta(days=1)

    batches = []  # one long DataFrame per batch of tickers
    size = config.DOWNLOAD_BATCH_SIZE  # how many tickers to download at once
    for start in range(0, len(tickers), size):  # start = 0, 100, 200, ...
        batch = tickers[start : start + size]  # the next (up to) 100 tickers
        print(f"Downloading tickers {start + 1} to {start + len(batch)}")

        raw = yf.download(  # returns a pandas table: one row per day
            batch,
            start=config.START_DATE,
            end=end.isoformat(),
            auto_adjust=True,  # adjust prices for splits and dividends
            progress=False,  # no progress bar
        )

        # raw["Close"] has one column per ticker. Convert to Polars and
        # unpivot to long format: one row per (date, ticker).
        close = pl.from_pandas(raw["Close"].reset_index()).unpivot(
            index="Date", variable_name="ticker", value_name="close"
        )
        volume = pl.from_pandas(raw["Volume"].reset_index()).unpivot(
            index="Date", variable_name="ticker", value_name="volume"
        )
        # Put close and volume side by side, matching on date and ticker
        batches.append(close.join(volume, on=["Date", "ticker"]))

    prices = pl.concat(batches, how="vertical_relaxed")  # stack the batches into one table
    # "vertical_relaxed" lets Int64 and Float64 columns stack: a batch without
    # any missing volume comes back as integers, the others as floats.

    prices = prices.select(
        pl.col("Date").cast(pl.Date).alias("date"),  # datetime -> plain date
        pl.col("ticker"),
        pl.col("close"),
        pl.col("volume").cast(pl.Float64),  # one number type for volume
    ).sort("ticker", "date")  # order by stock, then by day

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)  # create data/raw/ if needed
    prices.write_parquet(config.RAW_PRICES_PATH)  # save so reruns don't download again
    return prices


# ---------------------------------------------------------------------------
# 3. Transform (one lazy Polars query)
# ---------------------------------------------------------------------------
def build_returns_pipeline(
    prices: pl.LazyFrame,
    sectors: pl.LazyFrame,
    missing_threshold: float = config.MISSING_THRESHOLD,
    drop_tickers: list[str] = config.DUPLICATE_SHARE_CLASSES,
) -> pl.LazyFrame:
    """Describe the steps from prices to daily returns. Nothing runs until .collect()."""
    return (
        # Remove duplicate share classes (keep one ticker per company)
        prices.filter(~pl.col("ticker").is_in(drop_tickers))
        # Add company and sector to every price row (only tickers found in both tables)
        .join(sectors, on="ticker", how="inner")
        # Per stock: how many days have a price, and how many trading days exist overall
        .with_columns(
            pl.col("close").count().over("ticker").alias("n_prices"),  # non-null closes per stock
            pl.col("date").n_unique().alias("n_days"),  # trading days in the whole dataset
        )
        # Share of trading days with no price, per stock
        .with_columns((1 - pl.col("n_prices") / pl.col("n_days")).alias("missing_share"))
        # Keep only stocks with at most 5% missing days
        .filter(pl.col("missing_share") <= missing_threshold)
        # pct_change compares each row with the row above, so sort days in order within each stock
        .sort("ticker", "date")
        # Daily return = close today / close yesterday - 1, computed separately for each stock
        .with_columns(pl.col("close").pct_change().over("ticker").alias("ret"))
        # The first day of each stock has no previous price, so its return is null: drop it
        .filter(pl.col("ret").is_not_null())
        # Keep only the columns we need
        .select("date", "ticker", "company", "sector", "close", "ret")
    )


# ---------------------------------------------------------------------------
# 4. Pivot to a stocks x days matrix
# ---------------------------------------------------------------------------
def pivot_returns(returns_long: pl.DataFrame) -> pl.DataFrame:
    """One row per ticker, one column per date, values = daily returns."""
    matrix = returns_long.pivot(on="date", index="ticker", values="ret")  # long -> wide
    date_columns = sorted(matrix.columns[1:])  # date columns in time order (column 0 is ticker)
    return matrix.select("ticker", *date_columns).sort("ticker")  # ticker first, then dates


def fill_nulls_with_zero(matrix: pl.DataFrame) -> pl.DataFrame:
    """Treat a missing return as 0% (no price move), so the matrix has no holes."""
    return matrix.fill_null(0.0)


# ---------------------------------------------------------------------------
# 5. Sanity-check helpers
# ---------------------------------------------------------------------------
def missing_report(
    prices: pl.DataFrame, sectors: pl.DataFrame, missing_threshold: float = config.MISSING_THRESHOLD
) -> pl.DataFrame:
    """One row per ticker: how much history is missing, and whether it was kept."""
    n_days = prices["date"].n_unique()  # number of trading days in the data
    first_day = prices["date"].min()  # first trading day in the data

    report = prices.group_by("ticker").agg(  # one row per ticker
        pl.col("close").count().alias("n_prices"),  # days with a price
        pl.col("date").filter(pl.col("close").is_not_null()).min().alias("first_price"),  # first priced day
    )
    report = report.with_columns(
        (1 - pl.col("n_prices") / n_days).alias("missing_share"),  # share of days without a price
    )
    report = report.with_columns(
        (pl.col("missing_share") <= missing_threshold).alias("kept"),  # True if the stock passes
    )
    report = report.with_columns(  # a short reason for each stock
        pl.when(pl.col("kept")).then(pl.lit("kept"))
        .when(pl.col("n_prices") == 0).then(pl.lit("no price data"))
        .when(pl.col("first_price") > first_day).then(pl.lit("prices start late (new listing or spin-off)"))
        .otherwise(pl.lit("gaps in price history"))
        .alias("reason")
    )
    report = report.join(sectors.select("ticker", "company", "sector"), on="ticker")  # add names
    return report.sort("missing_share", descending=True)  # worst first


def null_report(matrix: pl.DataFrame) -> pl.DataFrame:
    """Number of empty (null) cells per ticker in the matrix, only for tickers that have any."""
    long = matrix.unpivot(index="ticker", variable_name="date", value_name="ret")  # back to long
    nulls = long.filter(pl.col("ret").is_null())  # keep only the empty cells
    return nulls.group_by("ticker").agg(pl.len().alias("n_null_days"))  # count per ticker


def spot_check_return(
    prices: pl.DataFrame, returns_long: pl.DataFrame, ticker: str, n_days: int = 5
) -> pl.DataFrame:
    """Compare the pipeline's returns for one stock with a by-hand calculation."""
    stock = prices.filter(pl.col("ticker") == ticker).sort("date").head(n_days + 1)  # first days
    dates = stock["date"].to_list()  # as Python lists, so the maths is plain Python
    closes = stock["close"].to_list()

    rows = []  # one row per day, starting from day 2 (day 1 has no previous close)
    for i in range(1, len(closes)):
        rows.append(
            {
                "date": dates[i],
                "prev_close": closes[i - 1],
                "close": closes[i],
                "manual_ret": closes[i] / closes[i - 1] - 1,  # return by hand
            }
        )
    manual = pl.DataFrame(rows)

    pipeline = returns_long.filter(pl.col("ticker") == ticker).select("date", "ret")  # our returns
    check = manual.join(pipeline, on="date")  # put both side by side
    return check.with_columns((pl.col("manual_ret") - pl.col("ret")).abs().alias("difference"))


def extreme_returns(returns_long: pl.DataFrame, limit: float = config.EXTREME_RETURN) -> pl.DataFrame:
    """Daily returns bigger than `limit` (default 50%) up or down, to check by eye."""
    return returns_long.filter(pl.col("ret").abs() > limit)


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------
def save_processed(returns_long: pl.DataFrame, matrix: pl.DataFrame, sectors: pl.DataFrame) -> None:
    """Save the three outputs to data/processed/."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)  # create the folder if needed
    returns_long.write_parquet(config.RETURNS_LONG_PATH)  # one row per stock-day
    matrix.write_parquet(config.RETURNS_MATRIX_PATH)  # stocks x days
    kept = sectors.filter(pl.col("ticker").is_in(matrix["ticker"].to_list()))  # kept stocks only
    kept.write_parquet(config.SECTORS_PATH)


def run_pipeline(force_refresh: bool = False) -> pl.DataFrame:
    """Fetch (or load) the data, compute returns, build and save the matrix."""
    sectors = fetch_sectors(force_refresh)  # step 1
    prices = fetch_prices(sectors["ticker"].to_list(), force_refresh)  # step 2
    returns_long = build_returns_pipeline(prices.lazy(), sectors.lazy()).collect()  # step 3
    matrix = fill_nulls_with_zero(pivot_returns(returns_long))  # step 4
    save_processed(returns_long, matrix, sectors)  # save
    return matrix


if __name__ == "__main__":  # runs only with `python -m src.data`
    matrix = run_pipeline()
    print("Matrix shape:", matrix.shape)
