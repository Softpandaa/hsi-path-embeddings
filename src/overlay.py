"""The fund: an equity book, a cash sleeve and an HSI derivatives overlay.

    NAV = E + C + V_opt
At each equity rebalance, c = m h beta with m = IM / (50 S) at IM_DATE and beta
the book's ex-ante beta to the HSI over the 252-day window; E is reset to
(1 - c) NAV and the transfer between cash and the book pays COST_BPS. Cash
accrues one-month HIBOR. Options are opened on listed expiries into the next
monthly contract, marked daily at Black-Scholes and settled at the close on
expiry. Every arm holds the same c, so arms differ only by their derivatives.

    P1, yield       short n calls at up(1.04 S), n = floor(h beta E / (50 S)), delta
                    hedged with H = round(-Delta / e^{(r-q) tau}) HSI futures reset
                    daily at the close; F = S e^{(r-q) tau} to the option expiry,
                    variation margin to cash; the futures expire with the options
    P2, protection  on when VHSI > 30 and off when VHSI < 22, acting at expiries:
                    long a put at down(0.95 S), short a put at down(0.90 S) and short
                    a call at up(K*), K* the zero net premium strike held within
                    [1.05 S, 1.10 S]; n as P1 and at most DELTA_BUDGET NAV / (50 S |delta|)
                    at entry
    diagnostics     "P1 unhedged" is P1 without its futures, "P2 Proposal 1" is P2 on
                    Proposal 1's full trigger, on when VHSI > 30 or the book's drawdown
                    from its running peak (1 included) exceeds 5%, off when VHSI < 22
                    and the drawdown is within 2%
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import config, metrics, options

M = config.MULTIPLIER
Q = config.DIV_YIELD


def margin_ratio(d):
    s = d["hsi"]["close"].loc[:config.IM_DATE].iloc[-1]
    return config.IM_HKD / (M * s)


def book_beta(d, pos, weights):
    """Ex-ante beta of the book to the HSI from the 252 daily log returns before pos."""
    cols = [d["tickers"].index(t) for t in weights.index]
    px = d["adj"].values[pos - config.WINDOW - 1:pos, cols]
    r = np.diff(np.log(px), axis=0)
    m = np.diff(np.log(d["hsi"]["close"].values[pos - config.WINDOW - 1:pos]))
    b = ((r - r.mean(0)) * (m - m.mean())[:, None]).sum(0) / ((m - m.mean()) ** 2).sum()
    return float(weights.values @ b)


def _tau(expiry, day):
    return max((expiry - day).days, 0) / 365.0


def _mark(pos, s, r, vhsi, day, shift):
    if pos is None:
        return 0.0
    t = _tau(pos["expiry"], day)
    return pos["n"] * M * sum(sg * options.black_scholes(kd, s, k, t, r, options.vol(kd, vhsi, shift))
                              for kd, k, sg in pos["legs"])


def _delta(pos, s, r, vhsi, day, shift):
    """Delta of the position in index units, contracts times delta."""
    if pos is None:
        return 0.0
    t = _tau(pos["expiry"], day)
    return pos["n"] * sum(sg * options.delta(kd, s, k, t, r, options.vol(kd, vhsi, shift))
                          for kd, k, sg in pos["legs"])


def _p2_legs(s, t, r, vhsi, shift):
    legs = [("put", options.down(config.PUT_LONG * s), 1), ("put", options.down(config.PUT_SHORT * s), -1)]
    spread = sum(sg * options.black_scholes(kd, s, k, t, r, options.vol(kd, vhsi, shift)) for kd, k, sg in legs)
    sig = options.vol("call", vhsi, shift)
    f = lambda k: options.black_scholes("call", s, k, t, r, sig) - spread
    lo, hi = config.CALL_BAND[0] * s, config.CALL_BAND[1] * s
    k3 = lo if f(lo) < 0 else (hi if f(hi) > 0 else brentq(f, lo, hi))
    return legs + [("call", options.up(k3), -1)]


def _open(pos, s, t, r, vhsi, shift, cash):
    for kd, k, sg in pos["legs"]:
        p = options.black_scholes(kd, s, k, t, r, options.vol(kd, vhsi, shift))
        cash -= sg * pos["n"] * M * p + pos["n"] * options.entry_cost(p)
    return cash


def run(d, book_returns, records, arm, h, shift=0.0):
    """Daily NAV and whether P2 is held, with the initial NAV, the beta at each
    rebalance, the (entry, expiry) of each P1 position and, at each P2 entry,
    whether the delta cap binds."""
    cal = d["cal"]
    spot, vix, rf, cash_lr = d["hsi"]["close"], d["vhsi"], d["rf"], d["cash"]
    expiry_days = options.expiries(cal)
    m = margin_ratio(d)
    rebal = {rec["date"]: rec for rec in records}
    start = cal.get_loc(records[0]["date"])
    p1, p2 = arm in ("P1", "P1+P2", "P1 unhedged"), arm in ("P2", "P1+P2", "P2 Proposal 1")
    hedged = arm != "P1 unhedged"
    wealth = (1 + book_returns.loc[cal[start + 1]:]).cumprod()
    drawdown = wealth / np.maximum(wealth.cummax(), 1.0) - 1

    nav0 = config.FUND_USD * d["usdhkd"].iloc[start]
    equity, cash, beta = 0.0, nav0, np.nan
    y = pr = None
    hedge, f_prev, on = 0, None, False
    info = {"nav0": nav0, "beta": [], "p1": [], "p2": []}
    rows = []
    for i in range(start, len(cal)):
        day = cal[i]
        s, v, r = spot.iloc[i], vix.iloc[i], rf.iloc[i]
        if i > start:
            equity *= 1 + book_returns.loc[day]
            cash *= np.exp(cash_lr.iloc[i])
        if hedge != 0:                                                    # futures variation margin
            f = s * np.exp((r - Q) * _tau(y["expiry"], day))
            cash += hedge * M * (f - f_prev)
            f_prev = f
        for pos in (y, pr):
            if pos is not None and day == pos["expiry"]:
                cash += _mark(pos, s, r, v, day, shift)                   # intrinsic value at the close
        if y is not None and day == y["expiry"]:
            info["p1"].append((y["entry"], y["expiry"]))
            y, hedge = None, 0
        if pr is not None and day == pr["expiry"]:
            pr = None
        if day in rebal:
            beta = book_beta(d, i, rebal[day]["weights"])
            info["beta"].append(beta)
            c = m * h * beta
            value = equity + cash + _mark(y, s, r, v, day, shift) + _mark(pr, s, r, v, day, shift)
            target = (1 - c) * value
            if i > start:                                                 # the first purchase is costed in the book
                cash -= abs(target - equity) * config.COST_BPS / 1e4
            cash -= target - equity
            equity = target
        if arm == "P2 Proposal 1":
            dd = drawdown.get(day, 0.0)
            on = (v > config.VHSI_ON or dd < -config.DD_ON) if not on else not (v < config.VHSI_OFF and dd >= -config.DD_OFF)
        else:
            on = v > config.VHSI_ON if not on else not v < config.VHSI_OFF
        later = expiry_days[expiry_days > day]
        if (day in expiry_days or i == start) and len(later):
            nxt = later[0]
            t = _tau(nxt, day)
            n = int(np.floor(h * beta * equity / (M * s)))
            if p2 and on and pr is None:
                legs = _p2_legs(s, t, r, v, shift)
                unit = abs(sum(sg * options.delta(kd, s, k, t, r, options.vol(kd, v, shift)) for kd, k, sg in legs))
                cap = int(np.floor(config.DELTA_BUDGET * (equity + cash + _mark(y, s, r, v, day, shift)) / (M * s * unit)))
                pr = {"expiry": nxt, "n": min(n, cap), "legs": legs}
                info["p2"].append(cap < n)
                cash = _open(pr, s, t, r, v, shift, cash)
            if p1 and y is None:
                y = {"expiry": nxt, "n": n, "legs": [("call", options.up(config.CALL_MONEYNESS * s), -1)], "entry": day}
                cash = _open(y, s, t, r, v, shift, cash)
                f_prev = s * np.exp((r - Q) * t)
        if y is not None and hedged:                                      # delta hedge reset at the close
            target_h = int(round(-_delta(y, s, r, v, day, shift) / np.exp((r - Q) * _tau(y["expiry"], day))))
            cash -= abs(target_h - hedge) * (config.FUTURES_FEES + M * config.FUTURES_HALF_SPREAD)
            hedge = target_h
        nav = equity + cash + (_mark(y, s, r, v, day, shift) + _mark(pr, s, r, v, day, shift))
        rows.append((day, nav, pr is not None))
    return pd.DataFrame(rows, columns=["date", "nav", "p2_on"]).set_index("date"), info


def nav_returns(nav, nav0):
    """Daily simple returns of the fund from the day after the start, the first
    measured from the initial NAV so that costs paid at the first close count."""
    v = nav["nav"]
    r = v.pct_change().iloc[1:]
    r.iloc[0] = v.iloc[1] / nav0 - 1
    return r


def implied_realised(d, positions):
    """VHSI at each P1 entry and the realised volatility of the HSI log return to the
    expiry, annualised by trading days per year, both in per cent."""
    close = d["hsi"]["close"]
    n = metrics.per_year(d["cal"][d["cal"].get_loc(positions[0][0]):])
    rows = []
    for entry, expiry in positions:
        r = np.diff(np.log(close.loc[entry:expiry].values))
        rows.append((entry, expiry, d["vhsi"].loc[entry], 100 * np.sqrt(n * np.mean(r ** 2))))
    return pd.DataFrame(rows, columns=["entry", "expiry", "implied", "realised"])


def breakeven(d, book_returns, records, arm, h, cash):
    """The shift k, in vol points, at which the arm's annual return equals that of no
    options; NaN when the arm does not beat no options at k = 0."""
    def ret(a, k):
        nav, info = run(d, book_returns, records, a, h, shift=k)
        return metrics.summary(nav_returns(nav, info["nav0"]), cash)["return"]
    base = ret("none", 0.0)
    gap = lambda k: ret(arm, k) - base
    if gap(0.0) <= 0:
        return np.nan
    if gap(config.SHIFT_MAX) > 0:
        return np.inf
    return brentq(gap, 0.0, config.SHIFT_MAX, xtol=0.01)
