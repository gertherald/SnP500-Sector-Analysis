"""Clustering: group stocks by how they move, without using their sector.

1. returns_by_sector()          returns as a numpy array, rows sorted by sector
2. standardise_rows()           each stock: mean 0, std 1 over its days
3. fit_pca()                    PCA scores of every stock
4. market_check()               is the first component the market? centred vs uncentred PCA
5. shuffled_variance()          PCA on shuffled data: how big do components get by chance?
6. count_real_components()      how many real components beat the shuffled ones
7. sector_mean_scores()         average PC1 / PC2 score per sector
8. fit_kmeans(), silhouette_sweep()   KMeans for many k, silhouette score for each
9. save_clusters()              write cluster labels and scores to Parquet
10. plots                       scree plot, sector scores, silhouette sweep
"""

import matplotlib.pyplot as plt  # static plots
import numpy as np  # maths on arrays
import polars as pl  # data processing
from sklearn.cluster import KMeans  # k-means clustering
from sklearn.decomposition import PCA  # principal component analysis
from sklearn.metrics import silhouette_samples, silhouette_score  # how well separated groups are

from src import config  # seeds and paths
from src.baseline import BLUE, GREY, ORANGE  # same colours as the baseline charts

GRID = "#dddddd"  # light grid lines
INK = "#191919"  # dark text and reference lines


# ---------------------------------------------------------------------------
# 1. Returns as a numpy array, sorted by sector
# ---------------------------------------------------------------------------
def returns_by_sector(
    matrix: pl.DataFrame, sectors: pl.DataFrame
) -> tuple[np.ndarray, pl.DataFrame]:
    """Stocks x days array of returns, rows sorted by sector then ticker.

    Returns the array and a table with the ticker and sector of each row.
    """
    # Add each stock's sector to its row, then sort so sectors sit together
    ordered = matrix.join(sectors.select("ticker", "sector"), on="ticker").sort("sector", "ticker")
    labels = ordered.select("ticker", "sector")  # row order, kept for later
    returns = ordered.drop("ticker", "sector").to_numpy()  # numbers only: stocks x days
    return returns, labels


# ---------------------------------------------------------------------------
# 2. Standardise each stock
# ---------------------------------------------------------------------------
def standardise_rows(returns: np.ndarray) -> np.ndarray:
    """Give every stock (row) a mean of 0 and a standard deviation of 1.

    After this, distance between two stocks only depends on their correlation,
    not on how volatile they are. (StandardScaler would scale the columns, the days.)
    """
    mean = returns.mean(axis=1, keepdims=True)  # each stock's average daily return
    std = returns.std(axis=1, keepdims=True)  # each stock's volatility
    return (returns - mean) / std  # subtract own mean, divide by own volatility


# ---------------------------------------------------------------------------
# 3. PCA
# ---------------------------------------------------------------------------
def fit_pca(z: np.ndarray, n_components: int) -> tuple[PCA, np.ndarray]:
    """Fit PCA on the standardised stocks x days matrix.

    Returns the fitted PCA object and the scores: one row per stock,
    one column per component.
    """
    # svd_solver="full": exact decomposition. With more than 500 columns sklearn would
    # otherwise switch to a faster randomized approximation that blurs later components.
    pca = PCA(n_components=n_components, svd_solver="full")
    scores = pca.fit_transform(z)  # sklearn centres each day column, then decomposes
    return pca, scores


# ---------------------------------------------------------------------------
# 4. Is the first component the market?
# ---------------------------------------------------------------------------
def market_check(z: np.ndarray, returns: np.ndarray) -> pl.DataFrame:
    """Compare the first component of centred PCA (sklearn) and uncentred SVD.

    For each version: how closely its time pattern follows the market
    (the equal-weighted average return per day), how many stocks sit on the
    same side of it, and how much variance it explains.
    """
    market = returns.mean(axis=0)  # average return of all stocks on each day

    # Version 1: sklearn PCA, which subtracts each day's average before decomposing
    pca, scores = fit_pca(z, n_components=1)
    centred_pattern = pca.components_[0]  # PC1 as a pattern over days
    centred_scores = scores[:, 0]  # each stock's PC1 score
    centred_share = pca.explained_variance_ratio_[0]  # share of variance explained

    # Version 2: SVD on z as it is, without centring the days
    u, s, vt = np.linalg.svd(z, full_matrices=False)  # z = u * s * vt
    uncentred_pattern = vt[0]  # first pattern over days
    uncentred_scores = u[:, 0] * s[0]  # each stock's score on it
    uncentred_share = s[0] ** 2 / (s**2).sum()  # share of the total sum of squares

    rows = []  # one row per version
    for name, pattern, stock_scores, share in [
        ("centred (sklearn PCA)", centred_pattern, centred_scores, centred_share),
        ("uncentred (SVD)", uncentred_pattern, uncentred_scores, uncentred_share),
    ]:
        corr_market = abs(np.corrcoef(pattern, market)[0, 1])  # sign of a component is arbitrary
        positive = (stock_scores > 0).mean()  # share of stocks with a positive score
        same_side = max(positive, 1 - positive)  # share on the bigger side
        rows.append(
            {
                "version": name,
                "corr_with_market": corr_market,
                "share_same_side": same_side,
                "variance_explained": share,
            }
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Noise benchmark: PCA on shuffled data
# ---------------------------------------------------------------------------
def shuffle_days(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle the order of days separately for every stock.

    Each stock keeps exactly the same returns, but the days no longer line up
    between stocks, so any pattern stocks share is destroyed.
    """
    shuffled = np.empty_like(z)  # same shape as z, filled below
    for i in range(z.shape[0]):  # one stock at a time
        shuffled[i] = rng.permutation(z[i])  # its returns in a random order
    return shuffled


def shuffled_variance(z: np.ndarray, n_shuffles: int, n_components: int) -> np.ndarray:
    """Variance explained by each component, for many shuffled copies of z.

    Returns an array of shape (n_shuffles, n_components).
    """
    rng = np.random.default_rng(config.RANDOM_SEED)  # fixed seed: same shuffles every run
    results = []  # one row of explained variance per shuffle
    for _ in range(n_shuffles):
        shuffled = shuffle_days(z, rng)  # a copy with no shared patterns
        pca, _ = fit_pca(shuffled, n_components)  # PCA on pure chance
        results.append(pca.explained_variance_ratio_)  # how big components get by chance
    return np.array(results)


# ---------------------------------------------------------------------------
# 6. How many components are real?
# ---------------------------------------------------------------------------
def noise_threshold(shuffled: np.ndarray, percentile: float) -> np.ndarray:
    """For each component number, the given percentile over the shuffled runs."""
    return np.percentile(shuffled, percentile, axis=0)  # one value per component


def count_real_components(real: np.ndarray, threshold: np.ndarray) -> int:
    """Number of leading components that explain more variance than chance.

    Counts from component 1 and stops at the first one that does not beat
    its threshold (Horn's parallel analysis).
    """
    count = 0
    for real_value, noise_value in zip(real, threshold):  # component 1, 2, 3, ...
        if real_value <= noise_value:  # no better than shuffled data: stop
            break
        count += 1
    return count


# ---------------------------------------------------------------------------
# 7. Average score per sector
# ---------------------------------------------------------------------------
def sector_mean_scores(scores: np.ndarray, labels: pl.DataFrame) -> pl.DataFrame:
    """Average PC1 and PC2 score of the stocks in each sector."""
    table = labels.with_columns(
        pl.Series("pc1", scores[:, 0]),  # each stock's score on component 1
        pl.Series("pc2", scores[:, 1]),  # and on component 2
    )
    return (
        table.group_by("sector")  # one group per sector
        .agg(pl.col("pc1").mean(), pl.col("pc2").mean(), pl.len().alias("n_stocks"))
        .sort("pc1")  # lowest PC1 first
    )


# ---------------------------------------------------------------------------
# 8. KMeans and silhouette score for many k
# ---------------------------------------------------------------------------
def fit_kmeans(scores: np.ndarray, k: int) -> np.ndarray:
    """KMeans with k clusters on the PCA scores. Returns a cluster number per stock."""
    model = KMeans(n_clusters=k, n_init=config.KMEANS_N_INIT, random_state=config.RANDOM_SEED)
    return model.fit_predict(scores)  # fit, then the nearest centre for every stock


def silhouette_sweep(scores: np.ndarray, k_values: range) -> pl.DataFrame:
    """Run KMeans for every k and score how well separated the clusters are."""
    rows = []  # one row per k
    for k in k_values:
        cluster = fit_kmeans(scores, k)  # cluster number of each stock
        rows.append({"k": k, "silhouette": silhouette_score(scores, cluster)})
    return pl.DataFrame(rows)


def silhouette_per_sector(scores: np.ndarray, labels: pl.DataFrame) -> pl.DataFrame:
    """Average silhouette per sector, using the official sectors as the groups.

    Positive: the sector's stocks are closer to each other than to the nearest other sector.
    """
    per_stock = silhouette_samples(scores, labels["sector"].to_numpy())  # one value per stock
    table = labels.with_columns(pl.Series("silhouette", per_stock))
    return (
        table.group_by("sector")
        .agg(pl.col("silhouette").mean(), pl.len().alias("n_stocks"))
        .sort("silhouette", descending=True)  # best separated sector first
    )



# ---------------------------------------------------------------------------
# 9. Save results for the next notebook
# ---------------------------------------------------------------------------
def save_clusters(labels: pl.DataFrame, cluster: np.ndarray, scores: np.ndarray) -> pl.DataFrame:
    """Write ticker, sector, cluster and PCA scores to data/processed/clusters.parquet."""
    table = labels.with_columns(pl.Series("cluster", cluster))  # cluster number per stock
    for i in range(scores.shape[1]):  # one column per component: pc1, pc2, ...
        table = table.with_columns(pl.Series(f"pc{i + 1}", scores[:, i]))
    table.write_parquet(config.CLUSTERS_PATH)
    return table


# ---------------------------------------------------------------------------
# 10. Plots
# ---------------------------------------------------------------------------
def _clean_axes(ax):
    """Light grid behind the data and no top/right box edges."""
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def plot_scree(real: np.ndarray, threshold: np.ndarray, n_real: int):
    """Variance explained per component: real data vs the shuffled-data threshold."""
    x = np.arange(1, len(real) + 1)  # component numbers 1, 2, 3, ...

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, real * 100, color=BLUE, linewidth=2, marker="o", markersize=4, label="Real data")
    ax.plot(x, threshold * 100, color=ORANGE, linewidth=2, label="Shuffled data (95th percentile)")
    ax.axvline(n_real + 0.5, color=INK, linewidth=0.8, linestyle="--")  # where real drops below noise
    ax.text(n_real + 1.5, real[0] * 100 * 0.5, f"{n_real} components\nbeat chance", fontsize=9, color=INK)

    ax.set_yscale("log")  # PC1 is much bigger than the rest, a log scale shows both
    ax.set_xlabel("Component")
    ax.set_ylabel("Variance explained (%, log scale)")
    ax.set_title("Which components are more than chance?")
    ax.legend(frameon=False)
    _clean_axes(ax)
    fig.tight_layout()
    return fig


def plot_sector_scores(summary: pl.DataFrame):
    """Two dot plots side by side: average PC1 and PC2 score per sector."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, column, title in [
        (axes[0], "pc1", "Component 1"),
        (axes[1], "pc2", "Component 2"),
    ]:
        data = summary.sort(column)  # lowest at the bottom
        y = np.arange(data.height)  # one row per sector
        values = data[column].to_numpy()  # the average scores
        colours = [BLUE if v > 0 else ORANGE for v in values]  # which side of zero
        ax.hlines(y, 0, values, color=GREY, linewidth=2)  # thin stem from zero to the dot
        ax.scatter(values, y, color=colours, s=50, zorder=3)  # dot at the sector average
        ax.axvline(0, color=INK, linewidth=0.8)  # zero line
        ax.set_yticks(y, data["sector"], fontsize=9)
        ax.set_xlabel("Average score of the sector's stocks")
        ax.set_title(title)
        _clean_axes(ax)
        ax.grid(axis="y", visible=False)  # only vertical grid lines
    fig.suptitle("Where each sector sits on the first two components")
    fig.tight_layout()
    return fig


def plot_silhouette_sweep(sweep: pl.DataFrame, sector_silhouette: float, chosen_k: int):
    """Silhouette score for each k, with the score of the official sectors for reference."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sweep["k"], sweep["silhouette"], color=BLUE, linewidth=2, marker="o", markersize=5,
            label="KMeans clusters")
    ax.axhline(sector_silhouette, color=ORANGE, linewidth=2, label="Official sectors (11 groups)")
    chosen = sweep.filter(pl.col("k") == chosen_k)["silhouette"][0]  # score at the chosen k
    ax.scatter([chosen_k], [chosen], s=140, facecolors="none", edgecolors=INK, linewidths=1.2,
               zorder=3, label=f"Chosen: k = {chosen_k}")  # ring around the chosen point

    ax.set_ylim(0, None)  # start at 0 so the gap to the sectors is honest
    ax.set_xticks(sweep["k"])
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("How well separated are the groups?")
    ax.legend(frameon=False, loc="lower right")
    _clean_axes(ax)
    fig.tight_layout()
    return fig
