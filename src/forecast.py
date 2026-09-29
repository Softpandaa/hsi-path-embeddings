"""The forecast of next-month relative return from an embedding, and its rank IC.

    y_{i,t} = sum_{s=1}^{21} r_{i,t+s} - r^f_{t,21},    u_{i,t} = y_{i,t} - mean_j y_{j,t},
    u_hat_{i,t} = a + b' m_{i,t},

with the mean over the members with a target on day t (relative to the cross
section, after Fischer and Krauss 2018) and (a, b) from pooled OLS on member
days whose 21-day target ends before the refit.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config

CACHE = Path(__file__).resolve().parents[1] / "cache"


def excess(d):
    """The target y, the 21-day forward excess log return, (days, stocks), NaN when
    any day is missing."""
    lr = np.log(d["adj"] / d["adj"].shift(1))
    h = config.HORIZON
    fwd = lr.rolling(h, min_periods=h).sum().shift(-h)
    rf = d["cash"].rolling(h, min_periods=h).sum().shift(-h)
    return fwd.sub(rf, axis=0)


def relative(d, y):
    """The relative target u, y less its mean over the members that have one on the
    same day."""
    return y.sub(y.where(d["member"]).mean(axis=1), axis=0)


def load_embedding(kind, year):
    z = np.load(CACHE / f"embedding_{kind}_{year}.npz")
    return z["day"], z["stock"], z["mu"]


class Forecaster:
    """The regression fitted at one refit and applied until the next."""

    def __init__(self, kind, year, refit_pos, u):
        self.day, self.stock, self.mu = load_embedding(kind, year)
        train = self.day <= refit_pos - config.HORIZON - 1                 # target ends before the refit
        obs = u.values[self.day[train], self.stock[train]]
        ok = np.isfinite(obs)
        X = np.column_stack([np.ones(ok.sum()), self.mu[train][ok]])
        self.coef, *_ = np.linalg.lstsq(X, obs[ok], rcond=None)
        pred = X @ self.coef
        self.r2 = float(1 - ((obs[ok] - pred) ** 2).sum() / ((obs[ok] - obs[ok].mean()) ** 2).sum())

    def predict(self, day):
        """Forecasts from the embedding of the given day, indexed by stock column."""
        rows = self.day == day
        return pd.Series(self.coef[0] + self.mu[rows] @ self.coef[1:], index=self.stock[rows])


def rank_ic(pred, realized):
    both = pd.concat([pred, realized], axis=1).dropna()
    return float(spearmanr(both.iloc[:, 0], both.iloc[:, 1]).statistic)
