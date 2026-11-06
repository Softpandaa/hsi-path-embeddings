# Price Path Embeddings in a Hang Seng Index Portfolio

This study asks whether embeddings of a stock's past year of trading carry predictive information, and what they deliver in a long-only HSI portfolio. Embeddings from principal component analysis (PCA) and an LSTM variational autoencoder (VAE) feed a pooled return forecast, Black-Litterman sets the weights, and listed HSI derivatives add delta-hedged short calls and a conditional put spread collar.

## Findings

From September 2021 to August 2026, the forecast has a rank information coefficient near 0.07 with a t-statistic near 2.5, and it reflects momentum over the past year. The VAE and PCA perform similarly, because the standardized return path behaves like a random walk.

The PCA portfolio with both derivatives strategies returns 6.28% a year against 3.09% for the HSI total return, an active return of 3.05% with an information ratio of 0.46. Stock selection adds 2.93% a year and the derivatives 0.98%, while portfolio construction alone loses 0.85%. Neither gain is statistically significant, although the forecast itself is.

## Layout

```
data/     committed prices, index membership and interest rates
src/      analysis modules, every parameter declared once in config.py
report.pdf
```

The report is distributed as a compiled PDF. Its typesetting source is not included.

## Data

`data/stocks.csv.gz` holds the daily prices, volumes, dividends and splits from Yahoo Finance of the stocks that were HSI members at some date between January 2005 and August 2026. `data/hsi_membership.csv` gives the inclusion and removal dates of each member from the Hang Seng Indexes press releases.

`data/hsi.csv`, `data/vhsi.csv`, `data/tracker_2800.csv` and `data/usdhkd.csv` hold the HSI, the VHSI, the Tracker Fund of Hong Kong (2800.HK) and the USD/HKD rate from Yahoo Finance, and `data/hibor_fixing.csv` the daily HIBOR fixing of the Hong Kong Monetary Authority. Seven codes are dropped in `config.py`, four without Yahoo data and three that Yahoo now assigns to another company.

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

The first three commands compute the embeddings at each of the five annual refits, training the VAE on a GPU where available. `src.run` then backtests the portfolios and derivatives strategies, and `src.report` prints every number quoted in the report and draws its five figures.