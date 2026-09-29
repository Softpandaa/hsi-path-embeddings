"""Load the committed data and apply the agreed cleaning rules.

Rules: HSI calendar to END; dropped tickers removed; placeholder rows before a
stock's first trade removed; open, high, low and close are Yahoo's split
adjusted prices as delivered and the dividend and split adjusted close is used
for returns only, as in CRSP; volume recorded as zero is missing and carried
forward (half-day sessions, and suspensions for the volume feature only); a
suspended day is zero volume with an unchanged close and cannot be traded;
Yahoo errors (listed days and nonpositive adjusted closes) lose every price on
that day; USD/HKD quotes outside the band are replaced by the previous quote;
the risk-free rate is the one-month HIBOR fixing, accrued actual/365 between
trading days.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

DATA = Path(__file__).resolve().parents[1] / "data"


def membership(cal, tickers):
    m = pd.read_csv(DATA / "hsi_membership.csv", parse_dates=["in_date", "out_date"])
    mem = pd.DataFrame(False, index=cal, columns=tickers)
    for _, s in m[m.ticker.isin(tickers)].iterrows():
        end = s.out_date if pd.notna(s.out_date) else cal[-1] + pd.Timedelta(days=1)
        mem.loc[(cal >= s.in_date) & (cal < end), s.ticker] = True
    return mem


def load():
    hsi = pd.read_csv(DATA / "hsi.csv", parse_dates=["date"]).set_index("date").loc[:config.END]
    cal = hsi.index
    px = pd.read_csv(DATA / "stocks.csv.gz", parse_dates=["date"])
    px = px[~px.ticker.isin(config.DROPPED)]

    # placeholder rows before the first trade
    first = px[px.volume > 0].groupby("ticker").date.min()
    px = px[px.date >= px.ticker.map(first)]

    wide = {c: px.pivot(index="date", columns="ticker", values=c).reindex(cal)
            for c in ("open", "high", "low", "close", "adj_close", "volume")}
    tickers = list(wide["close"].columns)

    raw_vol = wide["volume"]
    suspended = (raw_vol == 0) & (wide["close"].diff() == 0)
    volume = raw_vol.mask(raw_vol == 0).ffill()

    error = wide["adj_close"] <= 0
    for t, d in config.YAHOO_ERRORS:
        if t in error:
            error.loc[pd.Timestamp(d), t] = True
    for c in ("open", "high", "low", "close", "adj_close"):
        wide[c] = wide[c].mask(error)

    fx = pd.read_csv(DATA / "usdhkd.csv", parse_dates=["date"]).set_index("date")["close"]
    lo, hi = config.USDHKD_BAND
    fx = fx.mask((fx < lo) | (fx > hi)).ffill().reindex(cal).ffill()

    hib = pd.read_csv(DATA / "hibor_fixing.csv", parse_dates=["end_of_day"]).set_index("end_of_day")["ir_1m"]
    rf = hib.reindex(cal.union(hib.index)).ffill().reindex(cal) / 100.0
    gap = pd.Series(cal, index=cal).diff().dt.days
    cash = np.log1p(rf.shift(1) * gap / 365.0).fillna(0.0)      # daily log return of cash

    vol_idx = pd.read_csv(DATA / "vhsi.csv", parse_dates=["date"]).set_index("date")["close"]
    tracker = pd.read_csv(DATA / "tracker_2800.csv", parse_dates=["date"]).set_index("date")["adj_close"]

    return {
        "cal": cal, "tickers": tickers, "hsi": hsi,
        "open": wide["open"], "high": wide["high"], "low": wide["low"],
        "close": wide["close"], "adj": wide["adj_close"], "volume": volume,
        "suspended": suspended, "error": error,
        "member": membership(cal, tickers),
        "usdhkd": fx, "rf": rf, "cash": cash, "vhsi": vol_idx.reindex(cal).ffill(),
        "tracker": tracker.reindex(cal).ffill(),
    }


def refit_dates(cal):
    return [cal[cal >= pd.Timestamp(d)][0] for d in config.REFITS]


def rebalance_dates(cal):
    """First trading day of each month in the backtest."""
    c = cal[(cal >= pd.Timestamp(config.BACKTEST_START)) & (cal <= pd.Timestamp(config.END))]
    return pd.DatetimeIndex(pd.Series(c, index=c).groupby([c.year, c.month]).first().values)
