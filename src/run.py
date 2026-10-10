"""Backtest of Sections 3.3 to 3.5 from the embeddings in cache/, saved to cache/results.pkl."""

import pickle

import numpy as np
import pandas as pd

from . import config, data, derivatives, forecast, portfolios

VAE3 = f"vae{config.LATENT_M3}"
EMBEDDINGS = ("vae", VAE3, "pca")  # VAE with d = LATENT and d = LATENT_M3, PCA
FORECAST_PORTFOLIOS = {"M2-PCA": "pca", "M2-VAE": "vae", "M3": VAE3}
PORTFOLIOS = ("EW", "M1", *FORECAST_PORTFOLIOS)
M2 = ("M2-PCA", "M2-VAE")


class Weights:
    """Target weights of every portfolio at a rebalance, sharing the universe, Sigma, Pi and M1."""

    def __init__(self, d, forecasters, refit_pos):
        self.d, self.forecasters, self.refit_pos = d, forecasters, refit_pos
        self.cache = {}

    def predict(self, kind, pos):
        """Forecasts u_hat of equation (4) at the rebalance in row pos, from the day before."""
        k = max(i for i, p in enumerate(self.refit_pos) if p <= pos)
        return self.forecasters[kind][k].predict(pos - 1)

    def columns(self, pos):
        return portfolios.universe(self.d, pos, self.predict(EMBEDDINGS[0], pos).index.values)

    def common(self, pos, cols, fixed):
        key = (pos, tuple(cols), tuple(sorted(fixed.items())))
        if key not in self.cache:
            sigma, pi = portfolios.covariance(self.d, pos, cols)
            if not (np.isfinite(sigma).all() and np.isfinite(pi).all()):
                raise ValueError(f"incomplete return window at row {pos}")
            w1, ok1 = portfolios.mean_variance(sigma, pi, fixed)
            if not ok1:
                raise RuntimeError(f"M1 did not converge at row {pos}")
            self.cache[key] = (sigma, pi, w1)
        return self.cache[key]

    def portfolio(self, name):
        def target(pos, cols, fixed):
            sigma, pi, w1 = self.common(pos, cols, fixed)
            rec = {"p": len(cols), "solver_ok": True}
            if name == "EW":
                return portfolios.equal_weight(len(cols), fixed), rec
            if name == "M1":
                return w1, rec
            u = self.predict(FORECAST_PORTFOLIOS[name], pos).reindex(cols).values
            w, ok = portfolios.mean_variance(sigma, portfolios.posterior(sigma, pi, u), fixed)
            if not ok:  # hold M1 if the solver fails
                w = w1
            rec["solver_ok"] = ok
            return w, rec
        target.columns = self.columns
        return target


def block_returns(d, dates):
    """Simple return of every stock from one rebalance close to the next."""
    cal, adj = d["cal"], d["adj"]
    pos = [cal.get_loc(t) for t in dates] + [len(cal) - 1]
    return [adj.iloc[pos[k + 1]] / adj.iloc[pos[k]] - 1 for k in range(len(dates))]


def main():
    d = data.load()
    cal = d["cal"]
    dates = data.rebalance_dates(cal)
    refits = data.refit_dates(cal)
    refit_pos = [cal.get_loc(r) for r in refits]
    u = forecast.relative(d, forecast.excess(d))
    forecasters = {kind: [forecast.Forecaster(kind, r.year, p, u)
                          for r, p in zip(refits, refit_pos)]
                   for kind in EMBEDDINGS}
    weights = Weights(d, forecasters, refit_pos)

    regressions = {k: [(f.coef, f.r2) for f in v] for k, v in forecasters.items()}
    out = {"dates": dates, "regressions": regressions}
    for name in PORTFOLIOS:
        r, rec = portfolios.run(d, dates, weights.portfolio(name))
        out[name] = {"returns": r, "records": rec}
        print(f"{name}: {len(rec)} rebalances, M1 held after solver failure "
              f"{sum(not x['solver_ok'] for x in rec)}")

    realized = block_returns(d, dates)
    tickers = np.array(d["tickers"])
    for kind in EMBEDDINGS:
        ics = []
        for k, t in enumerate(dates):
            pos = cal.get_loc(t)
            cols = weights.columns(pos)
            pred = weights.predict(kind, pos).reindex(cols)
            pred.index = tickers[cols]
            ics.append(forecast.rank_ic(pred, realized[k].reindex(tickers[cols])))
        out[f"ic_{kind}"] = pd.Series(ics, index=dates)

    # all but EW at COVERAGE, the M2 portfolios also with the variants and at COVERAGE_FULL
    out["derivatives"] = {}
    for name in PORTFOLIOS[1:]:
        runs = [(config.COVERAGE, sc) for sc in config.SCENARIOS]
        if name in M2:
            runs += ([(config.COVERAGE, sc) for sc in config.VARIANTS]
                     + [(config.COVERAGE_FULL, sc) for sc in config.SCENARIOS])
        for h, sc in runs:
            nav, info = derivatives.run(d, out[name]["returns"], out[name]["records"], sc, h)
            out["derivatives"][(name, h, sc)] = {"nav": nav, "info": info}
    out["breakeven"] = derivatives.breakeven(d, out["M2-PCA"]["returns"], out["M2-PCA"]["records"],
                                             "P1 and P2", config.COVERAGE)
    print(f"derivatives done, break-even shift of P1 and P2 on M2-PCA {out['breakeven']:.2f}")
    p1 = out["derivatives"][("M2-PCA", config.COVERAGE, "P1")]["info"]["p1"]
    out["implied_realized"] = derivatives.implied_realized(d, p1)
    out["tracker"] = d["tracker"].pct_change()
    out["cash"] = d["cash"]
    with open(data.CACHE / "results.pkl", "wb") as fh:
        pickle.dump(out, fh)
    print("saved cache/results.pkl")


if __name__ == "__main__":
    main()
