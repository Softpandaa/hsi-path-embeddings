"""The fund: the stock portfolio, cash and the HSI derivatives strategies.

    V = E + B + O
At each rebalance the cash share is phi = rho h beta, with rho = IM / (50 S) at
IM_DATE and beta the stock portfolio's ex-ante beta to the HSI over the 252-day
window; E is reset to (1 - phi) V and the transfer between cash and the stock
portfolio pays COST_BPS. Cash accrues one-month HIBOR. Options are opened on
listed expiries into the next monthly contract, marked daily at Black-Scholes and
settled at the close on expiry. Every scenario holds the same phi, so the
scenarios differ only by their derivatives.

    P1, yield       short n calls at up(1.04 S), n = floor(h beta E / (50 S)), delta
                    hedged with H = round(-Delta / e^{(r-q) theta}) HSI futures reset
                    daily at the close; F = S e^{(r-q) theta} to the option expiry,
                    variation margin to cash; the futures expire with the options
    P2, protection  on when VHSI > 30 and off when VHSI < 22, acting at expiries:
                    long a put at down(0.95 S), short a put at down(0.90 S) and short
                    a call at up(K*), K* the zero net premium strike held within
                    [1.05 S, 1.10 S]; n as P1 and at most DELTA_BUDGET V / (50 S |delta|)
                    at entry
    variants        "P1 unhedged" is P1 without its futures, "P2, full trigger" is P2
                    on the proposal's full trigger, on when VHSI > 30 or the stock
                    portfolio's drawdown from its running peak (1 included) exceeds 5%,
                    off when VHSI < 22 and the drawdown is within 2%
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


def portfolio_beta(d, pos, weights):
    """Ex-ante beta of the stock portfolio to the HSI from the 252 daily log returns before pos."""
    cols = [d["tickers"].index(t) for t in weights.index]
    px = d["adj"].values[pos - config.WINDOW - 1:pos, cols]
    r = np.diff(np.log(px), axis=0)
    m = np.diff(np.log(d["hsi"]["close"].values[pos - config.WINDOW - 1:pos]))
    b = ((r - r.mean(0)) * (m - m.mean())[:, None]).sum(0) / ((m - m.mean()) ** 2).sum()
    return float(weights.values @ b)


def _theta(expiry, day):
    """Time to expiry in years, actual/365."""
    return max((expiry - day).days, 0) / 365.0


def _mark(pos, s, r, vhsi, day, shift):
    if pos is None:
        return 0.0
    t = _theta(pos["expiry"], day)
    return pos["n"] * M * sum(sg * options.black_scholes(kd, s, k, t, r, options.vol(kd, vhsi, shift))
                              for kd, k, sg in pos["legs"])


def _delta(pos, s, r, vhsi, day, shift):
    """Delta of the position in index units, contracts times delta."""
    if pos is None:
        return 0.0
    t = _theta(pos["expiry"], day)
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


def run(d, stock_returns, records, scenario, h, shift=0.0):
    """Daily NAV and whether P2 is held, with the initial NAV, the beta at each
    rebalance, the (entry, expiry) of each P1 position and, at each P2 entry,
    whether the delta cap binds."""
    cal = d["cal"]
    spot, vix, rf, cash_lr = d["hsi"]["close"], d["vhsi"], d["rf"], d["cash"]
    expiry_days = options.expiries(cal)
    rho = margin_ratio(d)
    rebal = {rec["date"]: rec for rec in records}
    start = cal.get_loc(records[0]["date"])
    has_p1, has_p2 = "P1" in scenario, "P2" in scenario
    hedged = scenario != "P1 unhedged"
    wealth = (1 + stock_returns.loc[cal[start + 1]:]).cumprod()
    drawdown = wealth / np.maximum(wealth.cummax(), 1.0) - 1

    nav0 = config.FUND_USD * d["usdhkd"].iloc[start]
    equity, cash, beta = 0.0, nav0, np.nan
    call = collar = None
    hedge, f_prev, on = 0, None, False
    info = {"nav0": nav0, "beta": [], "p1": [], "p2": []}
    rows = []
    for i in range(start, len(cal)):
        day = cal[i]
        s, v, r = spot.iloc[i], vix.iloc[i], rf.iloc[i]
        if i > start:
            equity *= 1 + stock_returns.loc[day]
            cash *= np.exp(cash_lr.iloc[i])
        if hedge != 0:                                                    # futures variation margin
            f = s * np.exp((r - Q) * _theta(call["expiry"], day))
            cash += hedge * M * (f - f_prev)
            f_prev = f
        for pos in (call, collar):
            if pos is not None and day == pos["expiry"]:
                cash += _mark(pos, s, r, v, day, shift)                   # intrinsic value at the close
        if call is not None and day == call["expiry"]:
            info["p1"].append((call["entry"], call["expiry"]))
            call, hedge = None, 0
        if collar is not None and day == collar["expiry"]:
            collar = None
        if day in rebal:
            beta = portfolio_beta(d, i, rebal[day]["weights"])
            info["beta"].append(beta)
            phi = rho * h * beta
            value = equity + cash + _mark(call, s, r, v, day, shift) + _mark(collar, s, r, v, day, shift)
            target = (1 - phi) * value
            if i > start:                                                 # the first purchase is costed in the stock portfolio
                cash -= abs(target - equity) * config.COST_BPS / 1e4
            cash -= target - equity
            equity = target
        if scenario == "P2, full trigger":
            dd = drawdown.get(day, 0.0)
            on = (v > config.VHSI_ON or dd < -config.DD_ON) if not on else not (v < config.VHSI_OFF and dd >= -config.DD_OFF)
        else:
            on = v > config.VHSI_ON if not on else not v < config.VHSI_OFF
        later = expiry_days[expiry_days > day]
        if (day in expiry_days or i == start) and len(later):
            nxt = later[0]
            t = _theta(nxt, day)
            n = int(np.floor(h * beta * equity / (M * s)))
            if has_p2 and on and collar is None:
                legs = _p2_legs(s, t, r, v, shift)
                unit = abs(sum(sg * options.delta(kd, s, k, t, r, options.vol(kd, v, shift)) for kd, k, sg in legs))
                cap = int(np.floor(config.DELTA_BUDGET * (equity + cash + _mark(call, s, r, v, day, shift)) / (M * s * unit)))
                collar = {"expiry": nxt, "n": min(n, cap), "legs": legs}
                info["p2"].append(cap < n)
                cash = _open(collar, s, t, r, v, shift, cash)
            if has_p1 and call is None:
                call = {"expiry": nxt, "n": n, "legs": [("call", options.up(config.CALL_MONEYNESS * s), -1)], "entry": day}
                cash = _open(call, s, t, r, v, shift, cash)
                f_prev = s * np.exp((r - Q) * t)
        if call is not None and hedged:                                   # delta hedge reset at the close
            target_h = int(round(-_delta(call, s, r, v, day, shift) / np.exp((r - Q) * _theta(call["expiry"], day))))
            cash -= abs(target_h - hedge) * (config.FUTURES_FEES + M * config.FUTURES_HALF_SPREAD)
            hedge = target_h
        nav = equity + cash + (_mark(call, s, r, v, day, shift) + _mark(collar, s, r, v, day, shift))
        rows.append((day, nav, collar is not None))
    return pd.DataFrame(rows, columns=["date", "nav", "p2_on"]).set_index("date"), info


def nav_returns(nav, nav0):
    """Daily simple returns of the fund from the day after the start, the first
    measured from the initial NAV so that costs paid at the first close count."""
    v = nav["nav"]
    r = v.pct_change().iloc[1:]
    r.iloc[0] = v.iloc[1] / nav0 - 1
    return r


def implied_realized(d, positions):
    """VHSI at each P1 entry and the realized volatility of the HSI log return to the
    expiry, annualized by trading days per year, both in per cent."""
    close = d["hsi"]["close"]
    n = metrics.per_year(d["cal"][d["cal"].get_loc(positions[0][0]):])
    rows = []
    for entry, expiry in positions:
        r = np.diff(np.log(close.loc[entry:expiry].values))
        rows.append((entry, expiry, d["vhsi"].loc[entry], 100 * np.sqrt(n * np.mean(r ** 2))))
    return pd.DataFrame(rows, columns=["entry", "expiry", "implied", "realized"])


def breakeven(d, stock_returns, records, scenario, h, cash):
    """The shift kappa, in vol points, at which the scenario's annual return equals
    that of no derivatives."""
    def ret(sc, k):
        nav, info = run(d, stock_returns, records, sc, h, shift=k)
        return metrics.summary(nav_returns(nav, info["nav0"]), cash)["return"]
    base = ret("None", 0.0)
    return brentq(lambda k: ret(scenario, k) - base, 0.0, config.SHIFT_MAX, xtol=0.01)
