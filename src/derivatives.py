"""The fund V = E + B + O of Appendix C with the derivatives strategies of Section 3.5."""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import config, metrics, options, portfolios

M = config.MULTIPLIER
Q = config.DIV_YIELD


def margin_ratio(d):
    """rho of equation (13)."""
    s = d["hsi"]["close"].loc[:config.IM_DATE].iloc[-1]
    return config.IM_HKD / (M * s)


def portfolio_beta(d, pos, weights):
    """beta_k: ex-ante beta of the stock portfolio to the HSI over the window before row pos."""
    cols = [d["tickers"].index(t) for t in weights.index]
    r, m = portfolios.returns(d, pos, cols)
    b = ((r - r.mean(0)) * (m - m.mean())[:, None]).sum(0) / ((m - m.mean()) ** 2).sum()
    return float(weights.values @ b)


def _theta(expiry, day):
    """Time to expiry in years, actual/365."""
    return max((expiry - day).days, 0) / 365.0


def _price(legs, s, t, r, vhsi, shift):
    """Signed sum of the leg prices, in index points."""
    return sum(sg * options.black_scholes(kd, s, k, t, r, options.vol(kd, vhsi, shift))
               for kd, k, sg in legs)


def _unit_delta(legs, s, t, r, vhsi, shift):
    return sum(sg * options.delta(kd, s, k, t, r, options.vol(kd, vhsi, shift))
               for kd, k, sg in legs)


def _mark(pos, s, r, vhsi, day, shift):
    """Value of an option position in HK$."""
    if pos is None:
        return 0.0
    return pos["n"] * M * _price(pos["legs"], s, _theta(pos["expiry"], day), r, vhsi, shift)


def _delta(pos, s, r, vhsi, day, shift):
    """Delta of a position in index units, contracts times delta."""
    return pos["n"] * _unit_delta(pos["legs"], s, _theta(pos["expiry"], day), r, vhsi, shift)


def _p2_legs(s, t, r, vhsi, shift):
    """Long K1 put, short K2 put and short K3 call of P2."""
    legs = [("put", options.down(config.PUT_LONG * s), 1),
            ("put", options.down(config.PUT_SHORT * s), -1)]
    spread = _price(legs, s, t, r, vhsi, shift)
    sig = options.vol("call", vhsi, shift)

    def f(k):
        return options.black_scholes("call", s, k, t, r, sig) - spread

    lo, hi = config.CALL_BAND[0] * s, config.CALL_BAND[1] * s
    k3 = lo if f(lo) < 0 else (hi if f(hi) > 0 else brentq(f, lo, hi))
    return [*legs, ("call", options.up(k3), -1)]


def _open(pos, s, t, r, vhsi, shift, cash):
    """Cash after the premiums and entry costs of a new position."""
    for kd, k, sg in pos["legs"]:
        p = options.black_scholes(kd, s, k, t, r, options.vol(kd, vhsi, shift))
        cash -= sg * pos["n"] * M * p + pos["n"] * options.entry_cost(p)
    return cash


def _margin(cash, hedge, f_prev, call, s, r, day):
    """Cash after the futures variation margin, and the price the futures last settled at."""
    if hedge == 0:
        return cash, f_prev
    f = s * np.exp((r - Q) * _theta(call["expiry"], day))
    return cash + hedge * M * (f - f_prev), f


def _reset(cash, equity, target, costed):
    """Cash after the stock portfolio moves from equity to target, paying COST_BPS if costed."""
    if costed:
        cash -= abs(target - equity) * config.COST_BPS / 1e4
    return cash - (target - equity)


def _trigger(on, v, dd):
    """P2 state after a close with VHSI v; the drawdown dd is 0 outside the full trigger."""
    if on:
        return not (v < config.VHSI_OFF and dd >= -config.DD_OFF)
    return v > config.VHSI_ON or dd < -config.DD_ON


def run(d, stock_returns, records, scenario, h, shift=0.0):
    """Daily NAV and P2 flag, with info: nav0, beta_k, P1 (entry, expiry) and P2 cap binding."""
    cal = d["cal"]
    spot, vhsi, rf, cash_lr = d["hsi"]["close"], d["vhsi"], d["rf"], d["cash"]
    expiry_days = options.expiries(cal)
    rho = margin_ratio(d)
    rebal = {rec["date"]: rec for rec in records}
    start = cal.get_loc(records[0]["date"])
    has_p1, has_p2 = "P1" in scenario, "P2" in scenario
    hedged = scenario != "P1 unhedged"
    full = scenario == "P2, full trigger"
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
        s, v, r = spot.iloc[i], vhsi.iloc[i], rf.iloc[i]
        if i > start:
            equity *= 1 + stock_returns.loc[day]
            cash *= np.exp(cash_lr.iloc[i])
        cash, f_prev = _margin(cash, hedge, f_prev, call, s, r, day)
        if call is not None and day == call["expiry"]:  # settled at intrinsic value, futures too
            cash += _mark(call, s, r, v, day, shift)
            info["p1"].append((call["entry"], call["expiry"]))
            call, hedge = None, 0
        if collar is not None and day == collar["expiry"]:
            cash += _mark(collar, s, r, v, day, shift)
            collar = None
        if day in rebal:
            beta = portfolio_beta(d, i, rebal[day]["weights"])
            info["beta"].append(beta)
            phi = rho * h * beta
            value = (equity + cash + _mark(call, s, r, v, day, shift)
                     + _mark(collar, s, r, v, day, shift))
            target = (1 - phi) * value
            # the first purchase is costed in the stock portfolio
            cash = _reset(cash, equity, target, i > start)
            equity = target
        on = _trigger(on, v, drawdown.get(day, 0.0) if full else 0.0)
        later = expiry_days[expiry_days > day]
        if (day in expiry_days or i == start) and len(later):
            nxt = later[0]
            t = _theta(nxt, day)
            n = int(np.floor(h * beta * equity / (M * s)))
            if has_p2 and on and collar is None:
                legs = _p2_legs(s, t, r, v, shift)
                unit = abs(_unit_delta(legs, s, t, r, v, shift))
                value = equity + cash + _mark(call, s, r, v, day, shift)
                cap = int(np.floor(config.DELTA_BUDGET * value / (M * s * unit)))
                collar = {"expiry": nxt, "n": min(n, cap), "legs": legs}
                info["p2"].append(cap < n)
                cash = _open(collar, s, t, r, v, shift, cash)
            if has_p1 and call is None:
                strike = options.up(config.CALL_MONEYNESS * s)
                call = {"expiry": nxt, "n": n, "legs": [("call", strike, -1)], "entry": day}
                cash = _open(call, s, t, r, v, shift, cash)
                f_prev = s * np.exp((r - Q) * t)
        if call is not None and hedged:  # delta hedge reset at the close
            carry = np.exp((r - Q) * _theta(call["expiry"], day))
            target_h = round(-_delta(call, s, r, v, day, shift) / carry)
            cash -= abs(target_h - hedge) * (config.FUTURES_FEES + M * config.FUTURES_HALF_SPREAD)
            hedge = target_h
        o = _mark(call, s, r, v, day, shift) + _mark(collar, s, r, v, day, shift)
        rows.append((day, equity + cash + o, collar is not None))
    return pd.DataFrame(rows, columns=["date", "nav", "p2_on"]).set_index("date"), info


def nav_returns(nav, nav0):
    """Daily simple fund returns, the first from nav0 so that costs at the first close count."""
    v = nav["nav"]
    r = v.pct_change().iloc[1:]
    r.iloc[0] = v.iloc[1] / nav0 - 1
    return r


def implied_realized(d, positions):
    """VHSI at each P1 entry and realized HSI volatility to the expiry, both in percent a year."""
    close = d["hsi"]["close"]
    n = metrics.per_year(d["cal"][d["cal"].get_loc(positions[0][0]):])
    rows = []
    for entry, expiry in positions:
        r = np.diff(np.log(close.loc[entry:expiry].values))
        rows.append((entry, expiry, d["vhsi"].loc[entry], 100 * np.sqrt(n * np.mean(r ** 2))))
    return pd.DataFrame(rows, columns=["entry", "expiry", "implied", "realized"])


def breakeven(d, stock_returns, records, scenario, h):
    """Shift k in vol points at which the scenario returns as much as none (Section 5)."""
    def ret(sc, k):
        nav, info = run(d, stock_returns, records, sc, h, shift=k)
        return metrics.summary(nav_returns(nav, info["nav0"]), d["cash"])["return"]
    base = ret("None", 0.0)
    return brentq(lambda k: ret(scenario, k) - base, 0.0, config.SHIFT_MAX, xtol=0.01)
