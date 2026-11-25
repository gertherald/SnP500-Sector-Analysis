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

---
