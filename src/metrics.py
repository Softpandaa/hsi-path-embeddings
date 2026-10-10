"""Performance measures of Appendix D; returns are daily simple, cash a daily log return."""

import numpy as np


def per_year(index):
    """A of Appendix D: trading days per calendar year over the span of index."""
    return (len(index) - 1) / ((index[-1] - index[0]).days / 365.25)


def summary(r, cash):
    r = r.dropna()
    n = per_year(r.index)
    lr = np.log1p(r)
    ex = lr - cash.reindex(r.index)
    wealth = np.exp(lr.cumsum())
    peak = np.maximum(wealth.cummax(), 1.0)  # the starting wealth of 1 counts as a peak
    return {
        "return": 100 * (np.exp(lr.mean() * n) - 1),
        "volatility": 100 * lr.std() * np.sqrt(n),
        "sharpe": ex.mean() / ex.std() * np.sqrt(n),
        "max_drawdown": 100 * (wealth / peak - 1).min(),
    }


def relative(r, bench, cash):
    """Beta, IR, and active return and tracking error in percent a year, on daily log returns."""
    idx = r.dropna().index.intersection(bench.dropna().index)
    n = per_year(idx)
    y = np.log1p(r.loc[idx]) - cash.loc[idx]
    x = np.log1p(bench.loc[idx]) - cash.loc[idx]
    active = y - x
    te = active.std() * np.sqrt(n)
    return {"beta": np.cov(y, x)[0, 1] / x.var(), "active": 100 * active.mean() * n,
            "tracking_error": 100 * te, "information_ratio": active.mean() * n / te}
