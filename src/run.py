"""The backtest, September 2021 to August 2026, after the latents are in cache/.

    python -m src.vae      latents of the LSTM-VAE, d = 2 (GPU)
    python -m src.vae --latent 3    latents of the LSTM-VAE, d = 3 (GPU)
    python -m src.pca      latents of the PCA benchmark
    python -m src.run      books, forecasts, derivatives strategies -> cache/results.pkl
"""

import pickle

import numpy as np
import pandas as pd

from . import books, config, data, forecast, overlay

VAE3 = f"vae{config.LATENT_M3}"
KINDS = ("vae", VAE3, "pca")                  # forecast latents: VAE d = LATENT, VAE d = LATENT_M3, PCA k = LATENT
FORECAST_BOOKS = {"M2_pca": "pca", "M2_vae": "vae", "M3_vae": VAE3}


class Targets:
    """Weights of every book at a rebalance, sharing the universe, the covariance
    and M1 across books."""

    def __init__(self, d, forecasters, refit_pos):
        self.d, self.forecasters, self.refit_pos = d, forecasters, refit_pos
        self.cache = {}

    def head(self, kind, pos):
        k = max(i for i, p in enumerate(self.refit_pos) if p <= pos)
        return self.forecasters[kind][k]

    def columns(self, pos):
        pred = self.head(KINDS[0], pos).predict(pos - 1)
        return books.universe(self.d, pos, pred.index.values)

    def common(self, pos, cols, fixed):
        key = (pos, tuple(cols), tuple(sorted(fixed.items())))
        if key not in self.cache:
            sigma, pi = books.covariance(self.d, pos, cols)
            if not (np.isfinite(sigma).all() and np.isfinite(pi).all()):
                raise ValueError(f"incomplete return window at row {pos}")
            w1, ok1 = books.mean_variance(sigma, pi, fixed)
            if not ok1:
                raise RuntimeError(f"M1 did not converge at row {pos}")
            self.cache[key] = (sigma, pi, w1)
        return self.cache[key]

    def book(self, name):
        def target(k, pos, cols, fixed):
            sigma, pi, w1 = self.common(pos, cols, fixed)
            rec = {"p": len(cols), "solver_ok": True}
            if name == "EW":
                return books.equal_weight(len(cols), fixed), rec
            if name == "M1":
                return w1, rec
            kind = FORECAST_BOOKS[name]
            score = self.head(kind, pos).predict(pos - 1).reindex(cols).values
            w, ok = books.mean_variance(sigma, books.posterior(sigma, pi, score), fixed)
            if not ok:                                         # hold M1 if the solver fails
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
    y = forecast.relative(d, forecast.labels(d))
    heads = {kind: [forecast.Forecaster(kind, r.year, p, y) for r, p in zip(refits, refit_pos)] for kind in KINDS}
    targets = Targets(d, heads, refit_pos)

    out = {"dates": dates, "heads": {k: [(h.coef, h.r2) for h in v] for k, v in heads.items()}}
    names = ["EW", "M1"] + list(FORECAST_BOOKS)
    for name in names:
        r, rec = books.run(d, dates, targets.book(name))
        out[name] = {"returns": r, "records": rec}
        print(f"{name}: {len(rec)} rebalances, M1 held after solver failure "
              f"{sum(not x['solver_ok'] for x in rec)}")

    realised = block_returns(d, dates)
    tickers = np.array(d["tickers"])
    for kind in KINDS:
        ics = []
        for k, t in enumerate(dates):
            pos = cal.get_loc(t)
            cols = targets.columns(pos)
            pred = targets.head(kind, pos).predict(pos - 1).reindex(cols)
            pred.index = tickers[cols]
            ics.append(forecast.rank_ic(pred, realised[k].reindex(tickers[cols])))
        out[f"ic_{kind}"] = pd.Series(ics, index=dates)

    # every proposed book at the main coverage, the M2 books also with the diagnostics and at h = 1
    out["overlay"] = {}
    for name in ["M1"] + list(FORECAST_BOOKS):
        runs = [(config.COVERAGE, arm) for arm in config.ARMS]
        if name in ("M2_pca", "M2_vae"):
            runs += ([(config.COVERAGE, arm) for arm in config.DIAGNOSTIC_ARMS]
                     + [(config.COVERAGE_REF, arm) for arm in config.ARMS])
        for h, arm in runs:
            nav, info = overlay.run(d, out[name]["returns"], out[name]["records"], arm, h)
            out["overlay"][(name, h, arm)] = {"nav": nav, "info": info}
    out["breakeven"] = overlay.breakeven(d, out["M2_pca"]["returns"], out["M2_pca"]["records"], "P1+P2",
                                         config.COVERAGE, d["cash"])
    print(f"derivatives done, break-even shift of P1 and P2 on M2_pca {out['breakeven']:.2f}")
    out["implied_realised"] = overlay.implied_realised(d, out["overlay"][("M2_pca", config.COVERAGE, "P1")]["info"]["p1"])
    out["tracker"] = d["tracker"].pct_change()
    out["cash"] = d["cash"]
    with open(forecast.CACHE / "results.pkl", "wb") as fh:
        pickle.dump(out, fh)
    print("saved cache/results.pkl")


if __name__ == "__main__":
    main()
