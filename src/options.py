"""HSI options of Section 3.5 and Appendix C: expiries, strikes, Black-Scholes and entry cost."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config

Q = config.DIV_YIELD


def expiries(cal):
    """The listed monthly expiry, the second to last trading day of each month."""
    s = pd.Series(cal, index=cal)
    return pd.DatetimeIndex([g.iloc[-2] if len(g) >= 2 else g.iloc[-1]
                             for _, g in s.groupby([cal.year, cal.month])])


def _step(level):
    return config.STRIKE_STEP_LOW if level < config.STRIKE_SWITCH else config.STRIKE_STEP_HIGH


def up(level):
    """First listed strike at or above level, on the level's grid (20,000 lies on both)."""
    return np.ceil(level / _step(level)) * _step(level)


def down(level):
    """First listed strike at or below level."""
    return np.floor(level / _step(level)) * _step(level)


def vol(kind, vhsi, shift):
    """Pricing volatility of a call or put from VHSI in percent and the shift k in vol points."""
    return (vhsi - shift if kind == "call" else vhsi + shift) / 100.0


def _d1(s, k, t, r, sigma):
    return (np.log(s / k) + (r - Q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))


def black_scholes(kind, s, k, t, r, sigma):
    """Price in index points, with the dividend yield q."""
    if t <= 0 or sigma <= 0:
        return max(k - s, 0.0) if kind == "put" else max(s - k, 0.0)
    d1 = _d1(s, k, t, r, sigma)
    d2 = d1 - sigma * np.sqrt(t)
    if kind == "put":
        return k * np.exp(-r * t) * norm.cdf(-d2) - s * np.exp(-Q * t) * norm.cdf(-d1)
    return s * np.exp(-Q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)


def delta(kind, s, k, t, r, sigma):
    if t <= 0 or sigma <= 0:
        return float(s > k) if kind == "call" else -float(s < k)
    n = norm.cdf(_d1(s, k, t, r, sigma))
    return np.exp(-Q * t) * (n if kind == "call" else n - 1)


def entry_cost(price):
    """HK$ per contract: half the spread psi of equation (14) plus fees."""
    if price <= config.SPREAD_LIMIT:
        spread = max(config.SPREAD_MIN, config.SPREAD_PCT * price)
    else:
        spread = config.SPREAD_ABOVE
    return config.MULTIPLIER * spread / 2 + config.OPTION_FEES
