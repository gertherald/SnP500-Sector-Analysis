# Do Sectors Describe How Stocks Move?

## Research question
How well do official sectors describe how S&P 500 stocks actually move, which stocks behave unlike their sector, and how stable are these groupings?

## Data
1. Daily prices for current S&P 500 stocks over the last two years, from the Yahoo Finance API (yfinance).
2. The S&P 500 constituent list with each company's current GICS sector, scraped from Wikipedia.

Both are processed in Polars: merged on ticker, stocks with over 5% missing price history dropped (index membership changes over time), and prices converted to daily returns.

## Approach
As a baseline, I compare average correlations within and across sectors. Then I cluster stocks by return patterns with scikit-learn's KMeans, using PCA to reduce dimensions and the silhouette score to choose the number of clusters. To separate real groupings from noise, I rerun the clustering on 200 bootstrap samples and on 3- and 6-month rolling windows, parallelised with Python multiprocessing.

## New learning
Polars, scikit-learn (KMeans, PCA, silhouette score), bootstrap stability analysis, multiprocessing and Plotly, none covered in the course.

## Expected outcome
A co-clustering heatmap and sector ranking showing which sectors move as coherent groups and which do not, plus an interactive Plotly map highlighting companies that behave unlike their label.
