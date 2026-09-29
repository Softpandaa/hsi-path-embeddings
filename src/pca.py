"""The linear embedding: the scores on the first d principal components of the
cumulative return path, fitted at each refit on the same training sequences as
the VAE, with scores saved for the same days and the eigenvalues and loadings.

    python -m src.pca
"""

import numpy as np
import torch

from . import config, data, features, vae

CHUNK = 20_000


def paths(batch, t, j):
    out = []
    for s in range(0, len(t), CHUNK):
        out.append(batch(t[s:s + CHUNK], j[s:s + CHUNK])[1].numpy().astype(np.float64))
    return np.concatenate(out)


def fit(d, x, batch, refit_pos):
    tt, jt, _, _ = vae.training_split(d, x, refit_pos)
    c = paths(batch, tt, jt)
    mean = c.mean(axis=0)
    lam, vec = np.linalg.eigh(np.cov(c, rowvar=False))
    order = np.argsort(lam)[::-1]
    vec = vec[:, order[:config.LATENT]]
    return mean, lam[order], vec, len(tt)


def main():
    d = data.load()
    x = features.build(d)
    f = x.shape[-1]
    # the path uses only the return channel, standardized per sequence, so the
    # pooled feature scaling does not enter
    batch = vae.Batcher(x, np.zeros(f, np.float32), np.ones(f, np.float32), torch.device("cpu"))
    refits = data.refit_dates(d["cal"])
    positions = [d["cal"].get_loc(r) for r in refits] + [len(d["cal"])]
    vae.CACHE.mkdir(exist_ok=True)
    for k, r in enumerate(refits):
        pos, nxt = positions[k], positions[k + 1]
        mean, lam, vec, n = fit(d, x, batch, pos)
        t, j = vae.window_days(d, x, 0, nxt)
        scores = (paths(batch, t, j) - mean) @ vec
        np.savez_compressed(vae.CACHE / f"embedding_pca_{r.year}.npz", day=t, stock=j, mu=scores,
                            eigenvalues=lam, loadings=vec)
        share = lam[:config.LATENT].sum() / lam.sum()
        print(f"refit {r.date()}: n_train {n}, top eigenvalues {np.round(lam[:4], 1)}, "
              f"share of first {config.LATENT} {share:.3f}, scores saved {len(t)}")


if __name__ == "__main__":
    main()
