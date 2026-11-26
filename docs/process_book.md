# Process Book: Do Sectors Describe How Stocks Move?

Short entries after every work session: what I worked on, problems, what I learned, resources used.

---

### 2026-11-20 

**Focus:** Scope refinement

**Worked on:**

*Scope and planning*
- Refined the scope: Polars for data processing, plus a stability analysis (bootstrap and rolling windows) using multiprocessing
- Reprioritised the visuals around the research question and updated the proposal

**Problems:**

*Scope*
- The dataset is small, so Polars and multiprocessing are not needed for speed. I had to find where they add real value: understanding how Polars plans a query, and running many independent clustering runs in parallel.

**Learned:**

*Polars*
- There is no index: grouping and ordering are always written out explicitly.
- Expressions describe a column; contexts (`select`, `with_columns`, `filter`, `group_by`) decide what happens with it.
- Window functions (`.over`) run a calculation separately per group. Without it, returns on a stacked table leak from one stock into the next.
- Lazy queries are only planned until `.collect()`. Comparing `.explain()` before and after optimisation shows Polars skipping unused columns, dropping helper columns early and filtering while reading files, without changing the result.


### 2026-11-22

**Focus:** Data pipeline

**Worked on:**

*Data collection*
- Set up the project with a fixed date range and random seed so results are reproducible
- Scraped the S&P 500 list with sectors from Wikipedia using BeautifulSoup
- Downloaded two years of daily prices from Yahoo Finance and saved them locally

**Problems**

*Data collection*
- Wikipedia blocks requests that use Python's default User-Agent; sending a custom one fixed it.
- Wikipedia and Yahoo write some tickers differently (BRK.B vs BRK-B), so tickers had to be converted before downloading.
- Some company names came out with missing or extra spaces, because a table cell can mix a link with plain text. Reading the cell's full text as it appears on the page fixed it; checking the names by eye caught the problem.
- Downloaded batches could not be combined because one column was stored as whole numbers in some batches and decimals in others. pandas changes the type silently when values are missing; Polars refuses to mix types, which surfaced the issue.
- Re-downloading gave slightly different prices, because Yahoo recalculates dividend-adjusted prices over time. Saving the raw download is the only way to get exactly the same results.

**Learned:**

*Data handling*
- Always inspect scraped text, even when the scraper runs without errors.
- Cache raw downloads: external data sources can change over time.
- Check extreme values in context before deciding whether they are errors.

### 2026-11-23

**Focus:** Data pipeline

**Worked on:**
*Cleaning and transformation*
- Built one lazy Polars query that joins the two sources, removes stocks with too much missing history and computes daily returns
- Reshaped the returns into a stocks x days matrix, ready for clustering
- Added sanity checks and a notebook that explains each step

**Problems**
*Data quality*
- Missing data was simpler than expected: the few dropped stocks were all new listings or spin-offs, and the remaining stocks have complete histories.
- A couple of very large one-day returns looked suspicious. Checking the days after showed that the price stayed at the new level, so they are real events rather than data errors, and were kept.

*Code*
- A small demo picked the wrong row because shifting a column leaves an empty value in the first row, which a filter then silently dropped.

**Learned:**

*Analysis setup*
- scikit-learn treats rows as samples and columns as features, so the matrix is stocks x days to cluster stocks by how they move.

**Resources:**
- Polars user guide: expressions and contexts, window functions, lazy API and query plans
- Polars API docs: `pivot`, `unpivot`, `concat`
- BeautifulSoup documentation
- yfinance documentation
- Wikipedia: List of S&P 500 companies
- Python `concurrent.futures` and threadpoolctl documentation

### 2026-11-25

**Focus:** Baseline

**Worked on:**

*Correlation baseline*
- Computed the correlation between every pair of stocks and drew it as a heatmap, with stocks sorted by sector
- Compared, for each sector, the average correlation within the sector with the average correlation to all other stocks
- Ranked the sectors by how much more their stocks move together than with the rest of the market

**Problems:**

*Interpretation*
- Almost every pair of stocks is positively correlated, because the whole market tends to move together. Comparing within-sector correlation with zero would make every sector look coherent, so each sector is compared with its own cross-sector correlation instead.
- Companies listed with two share classes correlate almost perfectly with themselves, which inflated their sector's score. Kept one ticker per company and compared the results with and without the duplicates: the affected sector's score dropped, but the ranking stayed the same. A quick estimate by hand underestimated the effect, because removing a ticker also removes all its pairs with the other stocks.

*Visualisation*
- With around 500 stocks, individual tickers cannot be labelled on the heatmap. Labelling the sector blocks and drawing lines between them made the structure readable.

**Learned:**

*Correlation*
- Correlation ignores how much a stock moves and only measures whether two stocks move in the same direction on the same days.
- A market-wide effect sits under every pair of stocks; the interesting signal is how far a group rises above it. This is the same effect the first principal component is expected to capture later.

*Findings*
- Energy and Utilities are the most coherent sectors: their stocks share a common driver (energy prices, interest rates).
- Communication Services, Health Care and Industrials are the least coherent: they group very different businesses under one label.
- Sector size does not explain coherence: a large sector can be coherent and a small one can be loose.

**Resources:**
- numpy documentation: `corrcoef`
- matplotlib documentation: `imshow`, colormaps, horizontal bar charts

**Next:**
- Standardise returns and run PCA; check whether the first component is the market

### 2026-11-26

**Focus:** Clustering

**Worked on:**

*PCA*
- Standardised every stock to mean 0 and standard deviation 1 over its own days
- Ran PCA on the standardised returns and checked whether the first component is the market
- Used a shuffle test to decide how many components carry shared patterns
- Read the first two components by averaging the scores per sector

*Clustering*
- Ran KMeans for 2 to 20 clusters on the PCA scores and compared the silhouette scores
- Scored the official sectors with the same silhouette measure, overall and per sector

**Problems:**

*Standardising*
- On raw returns, a stock that copies another stock's moves at three times the size looked further away than an unrelated quiet stock. Distance mixed volatility with co-movement. Standardising each stock fixed it: distance then depends only on correlation.
- scikit-learn's standard scaler works on columns, which here are days, not stocks. Standardising the rows by hand was simpler to explain.

*PCA*
- I expected the first component to be the market, but it was only weakly related to it and split the stocks into two halves. scikit-learn centres every day before PCA, and subtracting each day's average return is the same as removing the market move. Without centring, the first component follows the market almost perfectly.
- A manual check of the PCA scores only matched scikit-learn for the first component. With more than 500 columns, scikit-learn quietly switches to a faster approximate method. Asking for the exact method made all components match and changed the number of components kept.

*Choosing the number of clusters*
- The silhouette score had no clear peak: it rose slowly and then flattened, and a different random seed moved it about as much as one extra cluster. I chose 11 clusters, the same number as sectors, so clusters and sectors can be compared like for like.


**Learned:**

*PCA*
- PCA finds the directions in which stocks differ the most. Keeping the most variance is the same as losing the least information.
- The stock-by-stock and day-by-day matrices give the same components; one side gives each stock's score, the other gives the pattern over time.
- With standardised stocks, PCA is essentially a decomposition of the correlation heatmap from the baseline.
- After removing the market, the biggest pattern is growth vs defensive: technology on one side, consumer staples, utilities and real estate on the other.

*Permutation vs bootstrap*
- Shuffling each stock's days destroys any shared pattern but keeps each stock's own returns. PCA on shuffled data shows how big components get by chance, which gives a threshold for real components. About half of the variance is shared; the rest is company-specific news, which cannot help grouping.
- A permutation breaks a relationship to test "real or luck?". A bootstrap keeps relationships and resamples whole days to test "how much does the result wobble?". Both resample, but answer different questions.

*Findings*
- Clusters found from the data separate stocks much better than the official sectors.
- Per sector, Energy and Utilities are clear groups, while Industrials and Materials are closer to other sectors than to themselves. This matches the correlation baseline, found with a different method.

**Resources:**
- scikit-learn documentation: `PCA` (including `svd_solver`), `KMeans`, `silhouette_score`, `silhouette_samples`
- numpy documentation: `linalg.svd`, `linalg.eigh`
- Horn's parallel analysis for choosing the number of components

**Next:**
- Cross-tabulate clusters against sectors and list stocks that sit outside their sector's cluster

---
