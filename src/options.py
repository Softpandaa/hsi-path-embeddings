"""HSI index options: listed expiries and strikes, Black-Scholes marks and deltas
with a dividend yield, and the entry cost.

    price     Black-Scholes at VHSI, r = one-month HIBOR, q = DIV_YIELD, T actual/365.
              VHSI is a model-free 30-day implied variance index, not the implied
              volatility of the traded strike; calls are priced at VHSI - k and puts
              at VHSI + k in vol points, k = 0 in the main case and the break-even k
              reported as the measure of this limitation
    strikes   listed strikes are 100 points apart below 20,000 and 200 points apart at
              or above (HKEX); up() is the first listed strike at or above a level and
              down() the first at or below
    cost      half the HKEX market-maker maximum spread plus fees at entry,
              spread max(30, 0.1 P) points for P <= 750 and 75 above; nothing at
              expiry, as the options are cash settled
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config


def expiries(cal):
    """The listed monthly expiry, the second to last trading day of each month."""
    s = pd.Series(cal, index=cal)
    return pd.DatetimeIndex([g.iloc[-2] if len(g) >= 2 else g.iloc[-1]
                             for _, g in s.groupby([cal.year, cal.month])])


def _step(level):
    return config.STRIKE_STEP_LOW if level < config.STRIKE_SWITCH else config.STRIKE_STEP_HIGH


def up(level):
    """20,000 lies on both grids, so rounding on the grid of the level gives the listed strike."""
    return np.ceil(level / _step(level)) * _step(level)


def down(level):
    return np.floor(level / _step(level)) * _step(level)


def vol(kind, vhsi, shift):
    """Pricing volatility of a call or put from VHSI in per cent and the shift k in vol points."""
    return (vhsi - shift if kind == "call" else vhsi + shift) / 100.0


def _d1(s, k, t, r, sigma, q):
    return (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))


def black_scholes(kind, s, k, t, r, sigma, q=config.DIV_YIELD):
    """Price in index points."""
    if t <= 0 or sigma <= 0:
        return max(k - s, 0.0) if kind == "put" else max(s - k, 0.0)
    d1 = _d1(s, k, t, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(t)
    if kind == "put":
        return k * np.exp(-r * t) * norm.cdf(-d2) - s * np.exp(-q * t) * norm.cdf(-d1)
    return s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)


def delta(kind, s, k, t, r, sigma, q=config.DIV_YIELD):
    if t <= 0 or sigma <= 0:
        return float(s > k) if kind == "call" else -float(s < k)
    n = norm.cdf(_d1(s, k, t, r, sigma, q))
    return np.exp(-q * t) * (n if kind == "call" else n - 1)


def entry_cost(price):
    """HK$ per contract: half the maximum quoted spread plus fees."""
    spread = max(config.SPREAD_MIN, config.SPREAD_PCT * price) if price <= config.SPREAD_LIMIT else config.SPREAD_ABOVE
    return config.MULTIPLIER * spread / 2 + config.OPTION_FEES
