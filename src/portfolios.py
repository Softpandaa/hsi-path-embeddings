"""Stock portfolios of Section 3.4: universe, covariance (Appendix B), Black-Litterman weights."""

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from . import config


def _window(pos):
    """Rows of the WINDOW + 1 closes before row pos, which give WINDOW daily returns."""
    return slice(pos - config.WINDOW - 1, pos)


def returns(d, pos, cols):
    """Daily log returns of the stocks cols and of the HSI over the window before row pos."""
    r = np.diff(np.log(d["adj"].values[_window(pos), cols]), axis=0)
    m = np.diff(np.log(d["hsi"]["close"].values[_window(pos)]))
    return r, m


def universe(d, pos, embedded):
    """Stock columns of U_k at the rebalance in row pos."""
    adj = d["adj"].values
    ok = np.isfinite(adj[_window(pos)]).all(axis=0) & np.isfinite(adj[pos])
    ok &= d["member"].values[pos] & np.isin(np.arange(adj.shape[1]), embedded)
    return np.nonzero(ok)[0]


def covariance(d, pos, cols):
    """Sigma_k of equations (10) to (12) and Pi_k of equation (5)."""
    r, m = returns(d, pos, cols)
    T, p = r.shape
    sd = r.std(axis=0, ddof=1)
    c = np.corrcoef(r, rowvar=False)
    lam, vec = np.linalg.eigh(c)
    lam, vec = lam[::-1], vec[:, ::-1]
    edge = (1 + np.sqrt(p / T)) ** 2
    k = int(min(max((lam > edge).sum(), 1), config.K_MAX))
    lc = lam.copy()
    lc[k:] = lam[k:].mean()
    cc = (vec * lc) @ vec.T
    s = np.sqrt(np.diag(cc))
    cc = cc / np.outer(s, s)
    sigma = config.HORIZON * cc * np.outer(sd, sd)
    cov_m = ((r - r.mean(axis=0)) * (m - m.mean())[:, None]).sum(axis=0) / (T - 1)
    pi = config.BL_DELTA * config.HORIZON * cov_m
    return sigma, pi


def posterior(sigma, pi, u):
    """mu^BL of equation (6) from the forecasts u."""
    u = np.where(np.isfinite(u), u - np.nanmean(u), 0.0)  # no forecast: the view is the prior
    zs = config.BL_ZETA * sigma
    omega = np.diag(np.diag(zs))
    return pi + zs @ np.linalg.solve(zs + omega, u)


def _bounds(n, fixed):
    b = [(0.0, config.WEIGHT_CAP)] * n
    for i, w in fixed.items():
        b[i] = (w, w)
    return b


def _solve(mu, sigma, w0, A, b, bounds):
    scale = np.mean(np.diag(sigma))
    res = minimize(lambda w: -(w @ mu - 0.5 * config.BL_DELTA * w @ sigma @ w) / scale, w0,
                   jac=lambda w: -(mu - config.BL_DELTA * sigma @ w) / scale,
                   method="SLSQP", bounds=bounds,
                   constraints=[{"type": "eq", "fun": lambda w: A @ w - b, "jac": lambda w: A}],
                   options={"maxiter": 1000, "ftol": 1e-14})
    w = np.clip(res.x, 0.0, None)
    ok = res.success and np.abs(A @ w - b).max() < 1e-6
    return w, ok


def mean_variance(sigma, mu, fixed):
    """w_k of equation (7), from equal weight or, if SLSQP fails, from an LP feasible point."""
    n = len(sigma)
    A = np.vstack([np.ones(n)] + [np.eye(n)[i] for i in fixed])
    b = np.array([1.0, *fixed.values()])
    bounds = _bounds(n, fixed)
    w, ok = _solve(mu, sigma, equal_weight(n, fixed), A, b, bounds)
    if not ok:
        lp = linprog(np.zeros(n), A_eq=A, b_eq=b, bounds=bounds, method="highs")
        if not lp.success:
            raise RuntimeError("the portfolio constraints are infeasible")
        w, ok = _solve(mu, sigma, lp.x, A, b, bounds)
    return w, ok


def equal_weight(n, fixed):
    w = np.full(n, 0.0)
    free = [i for i in range(n) if i not in fixed]
    for i, v in fixed.items():
        w[i] = v
    w[free] = (1.0 - sum(fixed.values())) / len(free)
    return w


def hold(d, cols, w, start_pos, end_pos):
    """Daily returns of w held from the close of start_pos to end_pos, and the drifted weights."""
    px = d["adj"].iloc[:end_pos + 1, cols].ffill().values[start_pos:]  # a gap keeps the last close
    growth = px[1:] / px[0]
    value = growth @ w
    prev = np.concatenate([[1.0], value[:-1]])
    drift = w * growth[-1] / value[-1]
    return value / prev - 1.0, drift


def run(d, dates, target):
    """Daily net returns and rebalance records of target(pos, cols, fixed) -> (weights, record)."""
    cal = d["cal"]
    tickers = np.array(d["tickers"])
    index = {t: i for i, t in enumerate(tickers)}
    positions = [cal.get_loc(t) for t in dates] + [len(cal) - 1]
    held = pd.Series(dtype=float)
    rets, records = [], []
    for k, tau in enumerate(dates):
        pos, end = positions[k], positions[k + 1]
        susp = d["suspended"].values[pos]
        cols_all = target.columns(pos)
        frozen = {c: held[c] for c in held.index if susp[index[c]]}
        cols = np.array(sorted(set(cols_all) | {index[c] for c in frozen}))
        names = tickers[cols]
        # a suspended stock cannot be traded: if held it keeps its drifted weight, else zero
        fixed = {i: frozen.get(nm, 0.0) for i, nm in enumerate(names) if susp[index[nm]]}
        w, rec = target(pos, cols, fixed)
        new = pd.Series(w, index=names)
        both = new.index.union(held.index)
        turn = float(new.reindex(both, fill_value=0.0)
                     .sub(held.reindex(both, fill_value=0.0)).abs().sum())
        cost = turn * config.COST_BPS / 1e4
        r, drift = hold(d, cols, w, pos, end)
        if not np.isfinite(r).all():
            raise ValueError(f"missing prices in the block from {tau.date()}")
        r[0] = (1 + r[0]) * (1 - cost) - 1
        rets.append(pd.Series(r, index=cal[pos + 1:end + 1]))
        held = pd.Series(drift, index=names)
        held = held[held > 0]
        rec.update({"date": tau, "turnover": turn, "n": int((w > 1e-6).sum()),
                    "weights": new[new > 1e-9]})
        records.append(rec)
    return pd.concat(rets), records
