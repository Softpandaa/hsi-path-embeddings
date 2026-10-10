"""The committed data, cleaned as in Section 2."""

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
PRICES = ("open", "high", "low", "close", "adj_close")


def _read(name, date="date"):
    return pd.read_csv(DATA / name, parse_dates=[date]).set_index(date)


def membership(cal, tickers):
    m = pd.read_csv(DATA / "hsi_membership.csv", parse_dates=["in_date", "out_date"])
    mem = pd.DataFrame(False, index=cal, columns=tickers)
    for _, s in m[m.ticker.isin(tickers)].iterrows():
        end = s.out_date if pd.notna(s.out_date) else cal[-1] + pd.Timedelta(days=1)
        mem.loc[(cal >= s.in_date) & (cal < end), s.ticker] = True
    return mem


def load():
    hsi = _read("hsi.csv").loc[:config.END]
    cal = hsi.index
    px = pd.read_csv(DATA / "stocks.csv.gz", parse_dates=["date"])
    px = px[~px.ticker.isin(config.DROPPED)]

    # placeholder rows before the first trade
    first = px[px.volume > 0].groupby("ticker").date.min()
    px = px[px.date >= px.ticker.map(first)]

    wide = {c: px.pivot(index="date", columns="ticker", values=c).reindex(cal)
            for c in (*PRICES, "volume")}
    tickers = list(wide["close"].columns)

    raw_vol = wide["volume"]
    suspended = (raw_vol == 0) & (wide["close"].diff() == 0)
    volume = raw_vol.mask(raw_vol == 0).ffill()

    error = wide["adj_close"] <= 0
    for t, d in config.YAHOO_ERRORS:
        if t in error:
            error.loc[pd.Timestamp(d), t] = True
    for c in PRICES:
        wide[c] = wide[c].mask(error)

    fx = _read("usdhkd.csv")["close"]
    lo, hi = config.USDHKD_BAND
    fx = fx.mask((fx < lo) | (fx > hi)).ffill().reindex(cal).ffill()

    hib = _read("hibor_fixing.csv", "end_of_day")["ir_1m"]
    rf = hib.reindex(cal.union(hib.index)).ffill().reindex(cal) / 100.0
    gap = pd.Series(cal, index=cal).diff().dt.days
    cash = np.log1p(rf.shift(1) * gap / 365.0).fillna(0.0)  # r^f_t of Section 2

    vol_idx = _read("vhsi.csv")["close"]
    tracker = _read("tracker_2800.csv")["adj_close"]

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
