"""Every number and figure in the report, from cache/results.pkl and the embedding files in cache/.

    python -m src.report      prints the LaTeX table rows and the quoted statistics and writes latex/figures/
"""

import json
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, derivatives, forecast, metrics, run

FIG = Path(__file__).resolve().parents[1] / "latex" / "figures"
OKABE = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73", "orange": "#E69F00", "grey": "#666666",
         "purple": "#CC79A7"}
DASH = {"blue": (0, ()), "vermillion": (0, (5, 2)), "green": (0, (3, 1, 1, 1)), "orange": (0, (1, 1)),
        "grey": (0, (1, 2)), "purple": (0, (6, 1, 1, 1, 1, 1))}
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3})

PORTFOLIOS = ("EW", "M1") + tuple(run.FORECAST_PORTFOLIOS)
EMBEDDING_NAMES = {"pca": "PCA", "vae": f"VAE ($d={config.LATENT}$)", run.VAE3: f"VAE ($d={config.LATENT_M3}$)"}
ATTRIBUTION = [("Portfolio construction", "M1 against the HSI TR", "M1", "HSI"),
               ("Stock selection", "M2 against M1", "M2", "M1"),
               ("Derivatives, P1", "P1 against none", "P1", "None"),
               ("Derivatives, P2", "P1 and P2 against P1", "P1 and P2", "P1"),
               ("Total", "P1 and P2 against the HSI TR", "P1 and P2", "HSI")]


def tex(lines):
    """Table rows with typeset signs and no negative zero."""
    out = []
    for line in lines:
        if not line.startswith("%"):
            line = re.sub(r"(?<![\w.])[-+](0\.0+)(?!\d*[1-9])", r"\1", line)
            line = re.sub(r"(?<![\w.$])([-+])(?=\d)", lambda m: "$-$" if m.group(1) == "-" else "$+$", line)
        out.append(line)
    return "\n".join(out)


def load():
    with open(forecast.CACHE / "results.pkl", "rb") as fh:
        return pickle.load(fh)


def scenario_returns(out, name, scenario, h=config.COVERAGE):
    """Daily returns of a portfolio with the derivatives scenario at coverage h."""
    o = out["derivatives"][(name, h, scenario)]
    return derivatives.nav_returns(o["nav"], o["info"]["nav0"])


def active(r, ref, cash):
    """Active log return a year, tracking error, information ratio and its t."""
    a = metrics.relative(r, ref, cash)
    years = (r.index[-1] - r.index[0]).days / 365.25
    return a["active"], a["tracking_error"], a["information_ratio"], a["information_ratio"] * np.sqrt(years)


def portfolio_table(out):
    """Return, volatility, Sharpe ratio, max drawdown, turnover a month and mean names held."""
    cash, idx = out["cash"], out["EW"]["returns"].index
    lines = []
    for name in PORTFOLIOS:
        s, rec = metrics.summary(out[name]["returns"], cash), out[name]["records"]
        lines.append(f"{name} & {s['return']:.1f} & {s['volatility']:.1f} & {s['sharpe']:.2f} & {s['max_drawdown']:.1f} & "
                     f"{100 * np.mean([x['turnover'] for x in rec[1:]]):.1f} & {np.mean([x['n'] for x in rec]):.0f}" + r" \\")
    s = metrics.summary(out["tracker"].reindex(idx), cash)
    lines.append(f"HSI TR & {s['return']:.1f} & {s['volatility']:.1f} & {s['sharpe']:.2f} & "
                 f"{s['max_drawdown']:.1f} & & " + r" \\")
    return lines


def year_table(out):
    """Simple return over each September to August year in per cent, the portfolios as
    their return less that of the HSI total return."""
    idx = out["EW"]["returns"].index
    year = lambda r: np.expm1(np.log1p(r.fillna(0)).groupby(r.index.year - (r.index.month < 9)).sum())
    hsi = year(out["tracker"].reindex(idx))
    lines = [f"{k} & " + " & ".join(f"{100 * v:+.1f}" for v in year(out[k]["returns"]) - hsi) + r" \\"
             for k in PORTFOLIOS]
    return lines + ["HSI TR & " + " & ".join(f"{100 * v:.1f}" for v in hsi) + r" \\"]


def path_components(out):
    """The random walk eigenvectors sin((k - 1/2) pi j / L) and, per refit, the path
    eigenvalues, the loadings signed as the random walk eigenvectors, and the weight of
    the PCA forecast on the standardized return of each day of the window relative to
    the largest."""
    L = config.SEQ_LEN
    j = np.arange(1, L + 1)
    walk = np.stack([np.sin((k - 0.5) * np.pi * j / L) for k in (1, 2)], axis=1)
    rows = []
    for i, r in enumerate(config.REFITS):
        z = np.load(forecast.CACHE / f"embedding_pca_{r[:4]}.npz")
        sign = np.sign((z["loadings"] * walk).sum(axis=0))
        load = z["loadings"] * sign
        w = (np.cumsum(load[::-1], axis=0)[::-1] / np.sqrt(L)) @ (out["regressions"]["pca"][i][0][1:] * sign)
        rows.append((r[:4], z["eigenvalues"], load, w / w.max()))
    return walk, rows


def pca_lines(out):
    """The path components against those of a random walk, whose covariance min(i, j) / L
    has eigenvalues proportional to (k - 1/2)^-2, and the forecast weights on the window."""
    share = 100 / ((np.arange(1, 5) - 0.5) ** 2 * np.pi ** 2 / 2)
    lines = [f"% random walk: shares of the first four components {np.round(share, 1).tolist()}"]
    walk, rows = path_components(out)
    for y, lam, load, _ in rows:
        corr = [np.corrcoef(load[:, k], walk[:, k])[0, 1] for k in range(2)]
        lines.append(f"% {y}: shares {np.round(100 * lam[:4] / lam.sum(), 1).tolist()}, loading correlation with "
                     f"the random walk {np.round(corr, 4).tolist()}")
    first = [100 * lam[:config.LATENT].sum() / lam.sum() for _, lam, _, _ in rows]
    lines.append(f"% first {config.LATENT} path components carry {min(first):.1f}% to {max(first):.1f}% of the path variance")
    w = np.mean([row[-1] for row in rows], axis=0)
    blocks = [(0, 60), (60, 120), (120, 180), (180, 220), (220, 240)]
    lines.append("% forecast weight on the returns of days 1-60, 61-120, 121-180, 181-220, 221-240 of the window, "
                 "relative to the largest: " + ", ".join(f"{w[a:b].mean():.2f}" for a, b in blocks))
    return lines


def compare_lines(out):
    """M2-PCA and M2-VAE against each other."""
    act, _, ir, t = active(out["M2-PCA"]["returns"], out["M2-VAE"]["returns"], out["cash"])
    overlap = []
    for a, b in zip(out["M2-PCA"]["records"], out["M2-VAE"]["records"]):
        ix = a["weights"].index.union(b["weights"].index)
        overlap.append(1 - 0.5 * (a["weights"].reindex(ix, fill_value=0) - b["weights"].reindex(ix, fill_value=0)).abs().sum())
    return [f"% M2-PCA against M2-VAE: active {act:+.2f}, IR {ir:.2f}, t {t:.2f}; mean weight overlap {100 * np.mean(overlap):.0f}%"]


def forecast_table(out):
    """Rank IC by September to August year, mean, t and share positive; R2 of each
    forecast regression per refit."""
    lines, r2 = [], []
    for kind in run.FORECAST_PORTFOLIOS.values():
        ic = out[f"ic_{kind}"].dropna()
        t = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))
        by_year = ic.groupby(ic.index.year - (ic.index.month < 9)).mean()
        lines.append(f"{EMBEDDING_NAMES[kind]} & " + " & ".join(f"{v:.3f}" for v in by_year)
                     + f" & {ic.mean():.3f} & {t:.2f} & {100 * (ic > 0).mean():.0f}" + r" \\")
        r2.append(f"{EMBEDDING_NAMES[kind]} " + ", ".join(f"{100 * g[1]:.2f}" for g in out["regressions"][kind]))
    return lines + ["% forecast regression R2 per refit in per cent: " + "; ".join(r2)]


def embedding_table(out):
    """Per refit: training windows, active units of the VAE with d = 2 and d = 3, and the
    R2 of each dimension of the d = 2 embedding on the two PCA scores of the same windows."""
    lines = []
    for r in config.REFITS:
        y = r[:4]
        info = {k: json.load(open(forecast.CACHE / f"{k}_{y}.json")) for k in ("vae", run.VAE3)}
        v, p = forecast.load_embedding("vae", y)[2], forecast.load_embedding("pca", y)[2]
        X = np.column_stack([np.ones(len(p)), p])
        r2 = [1 - np.var(v[:, i] - X @ np.linalg.lstsq(X, v[:, i], rcond=None)[0]) / np.var(v[:, i])
              for i in range(v.shape[1])]
        lines.append(f"{y} & {info['vae']['n_train']:,} & {info['vae']['active_units']} & "
                     f"{info[run.VAE3]['active_units']} & " + " & ".join(f"{x:.2f}" for x in r2) + r" \\")
    lines.append(f"% rank IC correlation across months, VAE (d = {config.LATENT}) and PCA: "
                 f"{out['ic_vae'].corr(out['ic_pca']):.2f}")
    return lines


def active_table(out):
    """Against the HSI total return: beta, active return a year, tracking error, IR and
    t; against M1: active return, IR and t."""
    cash, idx = out["cash"], out["EW"]["returns"].index
    hsi, m1 = out["tracker"].reindex(idx), out["M1"]["returns"]
    lines = []
    for k in PORTFOLIOS:
        r = out[k]["returns"]
        act, te, ir, t = active(r, hsi, cash)
        vs = "& &" if k in ("EW", "M1") else "{:+.1f} & {:+.2f} & {:+.2f}".format(*np.array(active(r, m1, cash))[[0, 2, 3]])
        lines.append(f"{k} & {metrics.relative(r, hsi, cash)['beta']:.2f} & {act:+.1f} & {te:.1f} & {ir:+.2f} & "
                     f"{t:+.2f} & {vs}" + r" \\")
    return lines


def attribution_table(out):
    """Active return of the portfolio against the HSI total return split into steps,
    each against the one before: construction (M1), selection (M2), P1 and P2 given P1;
    active, IR and t for M2-PCA and M2-VAE. The rows add up to the total with the
    margin cash, none against M2, quoted in the caption."""
    cash, rows, margin, extra = out["cash"], {}, [], []
    for key in ("M2-PCA", "M2-VAE"):
        s = {sc: scenario_returns(out, key, sc) for sc in ("None", "P1", "P1 and P2")}
        idx = s["P1 and P2"].index
        s |= {"HSI": out["tracker"].reindex(idx), "M1": out["M1"]["returns"].reindex(idx), "M2": out[key]["returns"].reindex(idx)}
        for step in ATTRIBUTION:
            act, _, ir, t = active(s[step[2]], s[step[3]], cash)
            rows.setdefault(step[:2], []).append(f"{act:+.2f} & {ir:+.2f} & {t:+.2f}")
        margin.append(f"{key} {active(s['None'], s['M2'], cash)[0]:+.2f}")
        extra.append(f"{key} {active(s['P1 and P2'], s['None'], cash)[0]:+.2f}")
    return ([f"{a} & {b} & " + " & ".join(v) + r" \\" for (a, b), v in rows.items()]
            + ["% margin cash, none against M2: " + "; ".join(margin),
               "% derivatives, P1 and P2 against none: " + "; ".join(extra)])


def derivatives_table(out, h):
    """Per portfolio and scenario: return, volatility, Sharpe, max drawdown, return
    minus none and the t of the daily log difference, active return on the HSI total
    return and its information ratio, and the share of days P2 is held."""
    cash, tracker = out["cash"], out["tracker"]
    scenarios = config.SCENARIOS + (config.VARIANTS if h == config.COVERAGE else ())
    lines = []
    for key in ("M2-PCA", "M2-VAE"):
        rb = scenario_returns(out, key, "None", h)
        base = metrics.summary(rb, cash)["return"]
        for sc in scenarios:
            r = scenario_returns(out, key, sc, h)
            s = metrics.summary(r, cash)
            act, _, ir, _ = active(r, tracker.reindex(r.index), cash)
            gain = "" if sc == "None" else f"{s['return'] - base:+.2f} & {active(r, rb, cash)[3]:.2f}"
            days = f"{100 * out['derivatives'][(key, h, sc)]['nav']['p2_on'].mean():.0f}" if "P2" in sc else ""
            lines.append(f"{key if sc == 'None' else ''} & {sc} & {s['return']:.2f} & "
                         f"{s['volatility']:.1f} & {s['sharpe']:.2f} & {s['max_drawdown']:.1f} & "
                         + (gain or " & ") + f" & {act:+.2f} & {ir:.2f} & {days}" + r" \\")
    return lines


def derivatives_stats(out):
    """Statistics quoted in the text, on M2-PCA at the main coverage unless stated."""
    cash = out["cash"]
    get = lambda sc, h=config.COVERAGE: out["derivatives"][("M2-PCA", h, sc)]
    ret = lambda key, sc: metrics.summary(scenario_returns(out, key, sc), cash)["return"]
    keys = ["M1"] + list(run.FORECAST_PORTFOLIOS)
    gain = {sc: [ret(k, sc) - ret(k, "None") for k in keys] for sc in config.SCENARIOS[1:]}
    iv = out["implied_realized"]
    rel = get("P2")["nav"]["nav"] / get("None")["nav"]["nav"]
    top = rel.idxmax()
    low = rel.loc[top:].idxmin()
    lines = ["% gain over none across M1, M2-PCA, M2-VAE, M3: " + "; ".join(
                 f"{a} {min(g):+.2f} to {max(g):+.2f}" for a, g in gain.items()),
             f"% P1: VHSI above realized in {(iv['implied'] > iv['realized']).sum()} of {len(iv)} months",
             f"% P2 relative to none: peak {100 * (rel[top] - 1):+.1f}% on {top.date()}, then "
             f"{100 * (rel[low] - 1):+.1f}% on {low.date()}, end {100 * (rel.iloc[-1] - 1):+.1f}%"]
    on = get("P2")["nav"]["p2_on"]
    spells = on.ne(on.shift()).cumsum()[on]
    for _, days in spells.groupby(spells):
        a0, a1 = days.index[0], days.index[-1]
        lines.append(f"% P2 held {a0.date()} to {a1.date()}, relative NAV at the end {100 * (rel[a1] - 1):+.1f}%")
    e = get("P2", config.COVERAGE_FULL)["info"]["p2"]
    lines.append(f"% h = {config.COVERAGE_FULL}: delta cap binds at {sum(e)} of {len(e)} P2 entries")
    hb = config.COVERAGE * np.array(get("None")["info"]["beta"])
    lines.append(f"% h beta at the rebalances: mean {hb.mean():.3f}, range {hb.min():.3f} to {hb.max():.3f}")
    idx = get("None")["nav"].index[1:]
    lines.append(f"% HSI total return over the fund's dates {metrics.summary(out['tracker'].reindex(idx), cash)['return']:.2f}%")
    lines.append(f"% break-even shift of P1 and P2 in vol points: {out['breakeven']:.1f}")
    return lines


def _years(ax):
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))


def figure_portfolios(out):
    """Wealth of M1 and the M2 portfolios and the HSI total return, and the cumulative
    outperformance of every portfolio over the HSI total return."""
    idx = out["EW"]["returns"].index
    hsi = out["tracker"].reindex(idx)
    start = out["dates"][0]                                             # wealth 1 at the first rebalance close
    wealth = lambda r: pd.concat([pd.Series([1.0], index=[start]), (1 + r.fillna(0)).cumprod()])
    colors = {"EW": "grey", "M1": "blue", "M2-PCA": "green", "M2-VAE": "vermillion", "M3": "purple"}
    for name, keys, fn, ylabel in [("portfolios", ["M1", "M2-PCA", "M2-VAE"], wealth, "Wealth"),
                                   ("outperformance", list(colors), lambda r: 100 * (wealth(r) / wealth(hsi) - 1),
                                    "Outperformance over the HSI TR, %")]:
        fig, ax = plt.subplots(figsize=(8, 4))
        for k in keys:
            v = fn(out[k]["returns"])
            ax.plot(v.index, v.values, label=k, color=OKABE[colors[k]], linestyle=DASH[colors[k]], linewidth=1.4)
        if name == "portfolios":
            v = wealth(hsi)
            ax.plot(v.index, v.values, label="HSI TR", color=OKABE["orange"], linestyle=DASH["orange"], linewidth=1.4)
        ax.axhline(1.0 if name == "portfolios" else 0.0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_ylabel(ylabel)
        _years(ax)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(FIG / f"{name}.png", dpi=200)
        plt.close(fig)


def figure_loadings(out):
    """PC1 and PC2 loadings over the days of the window at every refit, with the random
    walk eigenvectors, and the weight of the PCA forecast on each day of the window."""
    L = config.SEQ_LEN
    j = np.arange(1, L + 1)
    walk, rows = path_components(out)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for k, ax in enumerate(axes[:2]):
        for n, row in enumerate(rows):
            ax.plot(j, row[2][:, k], color=OKABE["blue"], linewidth=1.0, alpha=0.6, label="Refits 2021 to 2025" if n == 0 else None)
        ax.plot(j, np.sqrt(2 / L) * walk[:, k], color="black", linestyle=(0, (4, 2)), linewidth=1.2, label="Random walk")
        ax.set_title(f"PC{k + 1} loading")
    for n, row in enumerate(rows):
        axes[2].plot(j, row[-1], color=OKABE["blue"], linewidth=1.0, alpha=0.6)
    axes[2].plot(j, np.mean([row[-1] for row in rows], axis=0), color=OKABE["vermillion"], linestyle=DASH["vermillion"],
                 linewidth=1.4, label="Mean")
    axes[2].set_title("Forecast weight")
    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_xlabel("Day of the window")
    axes[0].legend(frameon=False)
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "loadings.png", dpi=200)
    plt.close(fig)


def figure_payoff():
    """Return at expiry of the HSI alone and with the P2 collar, per unit of the index at
    entry, with the call strike drawn at the middle of its band."""
    x = np.linspace(0.8, 1.2, 401)
    k1, k2, k3 = config.PUT_LONG, config.PUT_SHORT, float(np.mean(config.CALL_BAND))
    collar = np.maximum(k1 - x, 0) - np.maximum(k2 - x, 0) - np.maximum(x - k3, 0)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.axvspan(*config.CALL_BAND, color=OKABE["grey"], alpha=0.12, linewidth=0)
    ax.plot(x, 100 * (x - 1), label="HSI", color=OKABE["blue"], linestyle=DASH["blue"], linewidth=1.4)
    ax.plot(x, 100 * (x - 1 + collar), label="HSI with P2", color=OKABE["vermillion"], linestyle=DASH["vermillion"], linewidth=1.4)
    for k, name in ((k2, "$K_2$"), (k1, "$K_1$"), (k3, "$K_3$")):
        ax.axvline(k, color="black", linewidth=0.6, alpha=0.5)
        ax.text(k, 21, name, ha="center", va="bottom")
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("HSI at expiry over HSI at entry")
    ax.set_ylabel("Return at expiry, %")
    ax.set_ylim(-21, 24)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "payoff.png", dpi=200)
    plt.close(fig)


def figure_derivatives(out, key="M2-PCA"):
    """NAV with each derivatives scenario relative to the NAV without derivatives."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    base = out["derivatives"][(key, config.COVERAGE, "None")]["nav"]["nav"]
    for sc, col in {"P1": "blue", "P2": "vermillion", "P1 and P2": "green"}.items():
        nav = out["derivatives"][(key, config.COVERAGE, sc)]["nav"]["nav"]
        ax.plot(nav.index, nav / base, label=sc, color=OKABE[col], linestyle=DASH[col], linewidth=1.4)
    ax.axhline(1.0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_ylabel("NAV relative to none")
    _years(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "derivatives.png", dpi=200)
    plt.close(fig)


def main():
    out = load()
    FIG.mkdir(parents=True, exist_ok=True)
    print(tex(pca_lines(out)))
    print(tex(compare_lines(out)))
    p = [x["p"] for x in out["M1"]["records"]]
    print(f"% portfolios: return, volatility, Sharpe, max drawdown, turnover a month, names; universe {min(p)} to {max(p)} stocks")
    print(tex(portfolio_table(out)))
    print("% portfolios: return less that of the HSI total return over each September to August year, and the HSI total return")
    print(tex(year_table(out)))
    print("% forecast: rank IC by September to August year, mean rank IC, t, share of months IC > 0")
    print(tex(forecast_table(out)))
    print("% embeddings: refit, training windows, active units d = 2 and d = 3, R2 of the d = 2 embedding on the PCA scores")
    print(tex(embedding_table(out)))
    print("% against the HSI total return: beta, active return, tracking error, IR, t; against M1: active, IR, t")
    print(tex(active_table(out)))
    for h in (config.COVERAGE, config.COVERAGE_FULL):
        print(f"% derivatives h = {h}: return, volatility, Sharpe, max drawdown, return minus none, t, active return "
              "on the HSI total return, IR, P2 days held")
        print(tex(derivatives_table(out, h)))
    print(tex(derivatives_stats(out)))
    print("% attribution at h = 0.5, each step against the one before: active, IR, t for M2-PCA and M2-VAE")
    print(tex(attribution_table(out)))
    figure_portfolios(out)
    figure_loadings(out)
    figure_payoff()
    figure_derivatives(out)
    print(f"figures written to {FIG}")


if __name__ == "__main__":
    main()
