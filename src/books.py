"""Stock books: the universe, the clipped covariance, and the Black-Litterman
books M1, M2 and M3 and equal weight.

At each rebalance tau, with information to the previous close and trades at the
close of tau, in one-month units (h = HORIZON trading days):
    universe  members with a price on tau, 252 complete returns before it, and a
              latent on the previous day
    Sigma     h times the sample covariance of the 252 daily log returns, correlation
              eigenvalues beyond the first K replaced by their average,
              K = min(#{lambda > (1 + sqrt(p/T))^2}, K_MAX), unit diagonal restored
    Pi        delta Sigma w_HSI = delta h cov(r, r_HSI), the equilibrium returns with
              the HSI as the market portfolio (He and Litterman 1999)
    M1        the prior alone, mu = Pi
    M2, M3    views that each stock beats its equilibrium return by its relative
              forecast, q = Pi + s - mean(s), with Omega = diag(tau Sigma), so
              mu = Pi + tau Sigma (tau Sigma + Omega)^{-1} (s - mean(s))
    both      max w'mu - delta/2 w'Sigma w  s.t.  1'w = 1, 0 <= w <= cap,
              a suspended stock keeps its drifted weight
Books are held without trading inside the month and pay COST_BPS one way on turnover.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from . import config


def universe(d, pos, latent_day):
    """Stock columns eligible at the rebalance in row pos."""
    adj = d["adj"].values
    window = adj[pos - config.WINDOW - 1:pos]                       # prices for 252 returns
    ok = np.isfinite(window).all(axis=0) & np.isfinite(adj[pos]) & d["member"].values[pos]
    ok &= np.isin(np.arange(adj.shape[1]), latent_day)
    return np.nonzero(ok)[0]


def covariance(d, pos, cols):
    """Clipped one-month covariance and the equilibrium returns Pi."""
    px = d["adj"].values[pos - config.WINDOW - 1:pos, cols]
    r = np.diff(np.log(px), axis=0)
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
    m = np.diff(np.log(d["hsi"]["close"].values[pos - config.WINDOW - 1:pos]))
    cov_m = ((r - r.mean(axis=0)) * (m - m.mean())[:, None]).sum(axis=0) / (T - 1)
    pi = config.BL_DELTA * config.HORIZON * cov_m
    return sigma, pi


def posterior(sigma, pi, score):
    """Expected returns with the views q = Pi + s - mean(s), P = I and
    Omega = diag(tau Sigma); tau cancels from the tilt."""
    s = np.where(np.isfinite(score), score, np.nan)
    s = np.where(np.isfinite(s), s - np.nanmean(s), 0.0)     # frozen names only; their weight is fixed
    ts = config.BL_TAU * sigma
    omega = np.diag(np.diag(ts))
    return pi + ts @ np.linalg.solve(ts + omega, s)


def _bounds(n, fixed):
    b = [(0.0, config.WEIGHT_CAP)] * n
    for i, w in fixed.items():
        b[i] = (w, w)
    return b


def _solve(mu, sigma, w0, A, b, bounds):
    scale = np.mean(np.diag(sigma))
    res = minimize(lambda w: -(w @ mu - 0.5 * config.BL_DELTA * w @ sigma @ w) / scale, w0,
                   jac=lambda w: -(mu - config.BL_DELTA * sigma @ w) / scale, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "eq", "fun": lambda w: A @ w - b, "jac": lambda w: A}],
                   options={"maxiter": 1000, "ftol": 1e-14})
    w = np.clip(res.x, 0.0, None)
    ok = res.success and np.abs(A @ w - b).max() < 1e-6
    return w, ok


def mean_variance(sigma, mu, fixed):
    """max w'mu - delta/2 w'Sigma w over the fully invested, capped, long-only books.
    Starts from equal weight and, if the solver fails, from an LP feasible point."""
    n = len(sigma)
    A = np.vstack([np.ones(n)] + [np.eye(n)[i] for i in fixed])
    b = np.array([1.0] + list(fixed.values()))
    bounds = _bounds(n, fixed)
    w, ok = _solve(mu, sigma, equal_weight(n, fixed), A, b, bounds)
    if not ok:
        lp = linprog(np.zeros(n), A_eq=A, b_eq=b, bounds=bounds, method="highs")
        if not lp.success:
            raise RuntimeError("the book constraints are infeasible")
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
    """Daily simple returns of a book bought at the close of start_pos and held to
    end_pos, and the weights it drifts to. Missing prices carry the last close."""
    px = d["adj"].iloc[:end_pos + 1, cols].ffill().values[start_pos:]
    growth = px[1:] / px[0]
    value = growth @ w
    prev = np.concatenate([[1.0], value[:-1]])
    drift = w * growth[-1] / value[-1]
    return value / prev - 1.0, drift


def run(d, dates, target):
    """Simulate one book. target(k, pos, cols, fixed) returns weights over cols and
    a record of the rebalance. Returns daily net returns and per-rebalance records."""
    cal = d["cal"]
    tickers = np.array(d["tickers"])
    positions = [cal.get_loc(t) for t in dates] + [len(cal) - 1]
    held = pd.Series(dtype=float)
    rets, records = [], []
    for k, tau in enumerate(dates):
        pos, end = positions[k], positions[k + 1]
        susp = d["suspended"].values[pos]
        cols_all = target.columns(pos)
        index = {t: i for i, t in enumerate(tickers)}
        frozen = {c: held[c] for c in held.index if susp[index[c]]}
        cols = np.array(sorted(set(cols_all) | {index[c] for c in frozen}))
        names = tickers[cols]
        # a suspended stock cannot be traded: held ones keep their drifted weight, others stay at zero
        fixed = {i: frozen.get(nm, 0.0) for i, nm in enumerate(names) if susp[index[nm]]}
        w, rec = target(k, pos, cols, fixed)
        new = pd.Series(w, index=names)
        turn = float(new.reindex(new.index.union(held.index), fill_value=0.0)
                     .sub(held.reindex(new.index.union(held.index), fill_value=0.0)).abs().sum())
        cost = turn * config.COST_BPS / 1e4
        r, drift = hold(d, cols, w, pos, end)
        if not np.isfinite(r).all():
            raise ValueError(f"missing prices in the block from {tau.date()}")
        r[0] = (1 + r[0]) * (1 - cost) - 1
        rets.append(pd.Series(r, index=cal[pos + 1:end + 1]))
        held = pd.Series(drift, index=names)
        held = held[held > 0]
        rec.update({"date": tau, "turnover": turn, "n": int((w > 1e-6).sum()), "weights": new[new > 1e-9]})
        records.append(rec)
    return pd.concat(rets), records
