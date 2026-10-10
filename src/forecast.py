"""The return forecast of Section 3.3, equations (3) and (4), and its rank IC."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config, data


def excess(d):
    """y of equation (3), (days, stocks), NaN when any day is missing."""
    lr = np.log(d["adj"] / d["adj"].shift(1))
    h = config.HORIZON
    fwd = lr.rolling(h, min_periods=h).sum().shift(-h)
    rf = d["cash"].rolling(h, min_periods=h).sum().shift(-h)
    return fwd.sub(rf, axis=0)


def relative(d, y):
    """u of equation (3), the mean taken over the members with a target on the day."""
    return y.sub(y.where(d["member"]).mean(axis=1), axis=0)


def load_embedding(kind, year):
    z = np.load(data.CACHE / f"embedding_{kind}_{year}.npz")
    return z["day"], z["stock"], z["mu"]


class Forecaster:
    """Regression (4) fitted at one refit and applied until the next."""

    def __init__(self, kind, year, refit_pos, u):
        self.day, self.stock, self.mu = load_embedding(kind, year)
        train = self.day <= refit_pos - config.HORIZON - 1  # target ends before the refit
        obs = u.values[self.day[train], self.stock[train]]
        ok = np.isfinite(obs)
        X = np.column_stack([np.ones(ok.sum()), self.mu[train][ok]])
        obs = obs[ok]
        self.coef, *_ = np.linalg.lstsq(X, obs, rcond=None)
        self.r2 = float(1 - ((obs - X @ self.coef) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())

    def predict(self, day):
        """Forecasts from the embedding of the given day, indexed by stock column."""
        rows = self.day == day
        return pd.Series(self.coef[0] + self.mu[rows] @ self.coef[1:], index=self.stock[rows])


def rank_ic(pred, realized):
    both = pd.concat([pred, realized], axis=1).dropna()
    return float(spearmanr(both.iloc[:, 0], both.iloc[:, 1]).statistic)
