# Price Path Embeddings in a Hang Seng Index Portfolio

This study explores whether embeddings of the recent trading history of a stock carry predictive information, and what they deliver in a portfolio. For each HSI stock, the daily return path over the past year and technical indicators are embedded in two dimensions, linearly by principal component analysis (PCA) and nonlinearly by a variational autoencoder (VAE) with LSTM layers. A pooled regression maps the embedding to a forecast of next-month returns. In a USD 100 million long-only portfolio, the forecast enters the Black-Litterman model to construct the weights, and listed HSI derivatives enhance yield through delta-hedged short calls and a put spread collar held at high volatility.

## Findings

From September 2021 to August 2026, the forecast has a rank information coefficient near 0.07 with a t-statistic near 2.5, and it amounts to momentum over the past year. The VAE and PCA perform similarly, because the standardized return path behaves like a random walk that leaves little structure beyond its principal components. The PCA portfolio with both derivatives strategies returns 6.28% a year against 3.09% for the HSI total return, an active return of 3.05% with an information ratio of 0.46. Stock selection adds 2.93% a year and the derivatives 0.98%, while portfolio construction alone loses 0.85%. Although the forecast itself is significant, neither its transfer into active return nor the gain from the derivatives is. 

## Layout

```
data/     committed prices, index membership and interest rates
src/      analysis modules, every parameter declared once in config.py
report.pdf
```

The report is distributed as a compiled PDF. Its typesetting source is not included.

## Data

`data/stocks.csv.gz` holds the daily open, high, low and close, adjusted close, volume, dividends and splits from Yahoo Finance for the stocks that were HSI members at some date between January 2005 and August 2026. `data/hsi_membership.csv` gives the inclusion and removal dates of each member from the Hang Seng Indexes press releases. `data/hsi.csv`, `data/vhsi.csv`, `data/tracker_2800.csv` and `data/usdhkd.csv` are the HSI, the VHSI, the Tracker Fund of Hong Kong (2800.HK) and the USD/HKD rate from Yahoo Finance, and `data/hibor_fixing.csv` is the daily HIBOR fixing of the Hong Kong Monetary Authority. Seven codes are dropped in `config.py`, four without Yahoo data and three that Yahoo now assigns to another company.

## Reproducing

Python 3.13.

```
pip install -r requirements.txt
python -m src.vae
python -m src.vae --latent 3
python -m src.pca
python -m src.run
python -m src.report
```

The first three commands write the embeddings at each of the five annual refits to `cache/`, with the VAE trained on a GPU where available. `src.run` backtests the portfolios and derivatives strategies into `cache/results.pkl`, and `src.report` prints every number quoted in the report and writes the five figures it uses to `latex/figures/`, which is created on first run and is not tracked.
