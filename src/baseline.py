"""Baseline: how strongly do stocks in the same sector move together?

1. load_processed()             read the returns matrix and sector labels
2. correlation_by_sector()      correlation between every pair of stocks, ordered by sector
3. sector_summary()             average correlation within each sector vs with other sectors
4. plot_correlation_heatmap()   heatmap with sector blocks
5. plot_sector_summary()        bar chart: within vs cross-sector correlation per sector
"""

import matplotlib.pyplot as plt  # static plots
import numpy as np  # fast maths on arrays (correlation)
import polars as pl  # data processing
from matplotlib.colors import LinearSegmentedColormap  # to build our own colour scale

from src import config  # file paths

# Colours: blue for "within sector", orange for "other sectors", light grey for zero
BLUE = "#257de0"
ORANGE = "#f47230"
GREY = "#efefef"


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_processed() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read the stocks x days returns matrix and the sector of each stock."""
    matrix = pl.read_parquet(config.RETURNS_MATRIX_PATH)  # one row per stock
    sectors = pl.read_parquet(config.SECTORS_PATH)  # ticker, company, sector, sub_industry
    return matrix, sectors


# ---------------------------------------------------------------------------
# 2. Correlation matrix, ordered by sector
# ---------------------------------------------------------------------------
def correlation_by_sector(
    matrix: pl.DataFrame, sectors: pl.DataFrame
) -> tuple[np.ndarray, pl.DataFrame]:
    """Correlation of daily returns between every pair of stocks.

    Stocks are sorted by sector (then ticker), so stocks of the same sector sit
    next to each other. Returns the correlation array and the matching labels.
    """
    # Add each stock's sector to its row of returns, then sort by sector and ticker
    ordered = matrix.join(sectors.select("ticker", "sector"), on="ticker").sort("sector", "ticker")

    labels = ordered.select("ticker", "sector")  # the row order, kept for the plots
    returns = ordered.drop("ticker", "sector").to_numpy()  # numbers only: stocks x days

    # np.corrcoef treats each row as one variable: result is stocks x stocks,
    # where cell [i, j] is the correlation between stock i and stock j
    corr = np.corrcoef(returns)
    return corr, labels


# ---------------------------------------------------------------------------
# 3. Within-sector vs cross-sector correlation
# ---------------------------------------------------------------------------
def sector_summary(corr: np.ndarray, labels: pl.DataFrame) -> pl.DataFrame:
    """For each sector: average correlation with its own sector vs with all other stocks."""
    sector_of = labels["sector"].to_numpy()  # sector of each row/column of corr

    rows = []  # one dictionary per sector
    for sector in np.unique(sector_of):  # every sector name, alphabetically
        inside = sector_of == sector  # True for stocks in this sector
        n = inside.sum()  # number of stocks in this sector

        block = corr[inside][:, inside]  # correlations between stocks of this sector
        # Average without the diagonal (a stock with itself is always 1.0)
        within = (block.sum() - n) / (n * (n - 1))

        cross = corr[inside][:, ~inside].mean()  # this sector's stocks vs all other stocks

        rows.append({"sector": sector, "n_stocks": int(n), "within": within, "cross": cross})

    summary = pl.DataFrame(rows)
    summary = summary.with_columns((pl.col("within") - pl.col("cross")).alias("gap"))  # how much stronger
    return summary.sort("gap", descending=True)  # most coherent sector first


def average_correlation(corr: np.ndarray) -> float:
    """Average correlation over all pairs of different stocks (the diagonal is skipped)."""
    n = corr.shape[0]  # number of stocks
    return (corr.sum() - n) / (n * (n - 1))


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(corr: np.ndarray, labels: pl.DataFrame):
    """Heatmap of the correlation matrix with lines between sectors."""
    # Colour scale: orange (negative) -> grey (zero) -> blue (positive)
    cmap = LinearSegmentedColormap.from_list("orange_grey_blue", [ORANGE, GREY, BLUE])

    fig, ax = plt.subplots(figsize=(10, 9))  # one square-ish plot
    image = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)  # draw the matrix as coloured cells
    fig.colorbar(image, ax=ax, shrink=0.8, label="Correlation of daily returns")

    # Where each sector starts and ends in the sorted order
    sector_of = labels["sector"].to_list()
    starts = [i for i in range(len(sector_of)) if i == 0 or sector_of[i] != sector_of[i - 1]]
    ends = starts[1:] + [len(sector_of)]

    for start in starts[1:]:  # a thin line at every sector boundary
        ax.axhline(start - 0.5, color="#191919", linewidth=0.5)
        ax.axvline(start - 0.5, color="#191919", linewidth=0.5)

    middles = [(s + e - 1) / 2 for s, e in zip(starts, ends)]  # centre of each sector block
    names = [sector_of[s] for s in starts]  # sector name for each block
    ax.set_yticks(middles, names, fontsize=8)  # sector names on the left
    ax.set_xticks(middles, names, fontsize=8, rotation=90)  # and at the bottom

    ax.set_title("Correlation between S&P 500 stocks, sorted by sector")
    fig.tight_layout()
    return fig


def plot_sector_summary(summary: pl.DataFrame):
    """Horizontal bars: within-sector vs cross-sector average correlation for each sector."""
    data = summary.sort("gap")  # smallest gap at the bottom, largest at the top
    y = np.arange(data.height)  # one position per sector
    height = 0.38  # thickness of each bar

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y + height / 2, data["within"], height, color=BLUE, label="Within own sector")
    ax.barh(y - height / 2, data["cross"], height, color=ORANGE, label="With other sectors")

    ax.set_yticks(y, data["sector"])  # sector names
    ax.set_xlabel("Average correlation of daily returns")
    ax.set_title("Do stocks move more with their own sector?")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)  # light vertical grid lines
    ax.set_axisbelow(True)  # grid behind the bars
    for side in ["top", "right"]:  # remove the box edges we don't need
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig
