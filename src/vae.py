"""Pooled LSTM-VAE: one encoder and decoder shared by every stock, latent size d = 2
(config.LATENT, books M2) or d = 3 with --latent 3 (config.LATENT_M3, book M3).

Input is a stock's 240-day sequence of the sixteen features. The decoder
rebuilds the cumulative path of the standardised returns,
c_s = sum_{u <= s} r_u / sqrt(240), since the daily returns themselves are
close to white noise and a latent cannot compress them. Training stops early
on the final twelve months before each refit. After each refit the latent
means are saved for every member day before the refit, which train the
forecast, and for the twelve months that follow, which it is applied to.
Run on a GPU where available:

    python -m src.vae                every September refit in config.REFITS, d = LATENT
    python -m src.vae --latent 3     latent size 3 (files vae3_*, latent_vae3_*)
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS, set before CUDA starts

import numpy as np
import torch
import torch.nn as nn

from . import config, data, features

CACHE = Path(__file__).resolve().parents[1] / "cache"
L = config.SEQ_LEN


class LSTMVAE(nn.Module):
    def __init__(self, n_features, d):
        super().__init__()
        h, k, p = config.HIDDEN, config.LAYERS, config.DROPOUT
        self.enc = nn.LSTM(n_features, h, k, batch_first=True, dropout=p)
        self.drop = nn.Dropout(p)
        self.mu, self.logvar = nn.Linear(h, d), nn.Linear(h, d)
        self.dec = nn.LSTM(d, h, k, batch_first=True, dropout=p)
        self.out = nn.Linear(h, 1)

    def encode(self, x):
        _, (h, _) = self.enc(x)
        h = self.drop(h[-1])
        return self.mu(h), self.logvar(h)

    def decode(self, z, length):
        y, _ = self.dec(z.unsqueeze(1).expand(-1, length, -1))
        return self.out(self.drop(y)).squeeze(-1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z, x.size(1)), mu, logvar


def elbo_loss(rec, target, mu, logvar):
    """Per-sequence negative ELBO, Gaussian likelihood of variance 1/2 up to a constant."""
    recon = ((rec - target) ** 2).sum(dim=1)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    return (recon + kl).mean(), recon.mean(), kl.mean()


def window_days(d, x, start, end):
    """(day, stock) pairs with start <= day < end: member on the day, a complete
    240-day feature window ending on the day, and no Yahoo error inside it."""
    finite = np.isfinite(x).all(axis=-1)                                   # (days, stocks)
    window_ok = np.lib.stride_tricks.sliding_window_view(finite, L, axis=0).all(axis=-1)
    err = d["error"].values
    err_in = np.lib.stride_tricks.sliding_window_view(err, L, axis=0).any(axis=-1)
    ok = np.zeros_like(finite)
    ok[L - 1:] = window_ok & ~err_in
    ok &= d["member"].values
    ok[:start] = False
    ok[end:] = False
    return np.nonzero(ok)


def training_split(d, x, refit_pos):
    """Training and validation sequences at a refit, the validation set being
    the final twelve months before it. Shared with the PCA benchmark."""
    t, j = window_days(d, x, 0, refit_pos)
    valid = t >= refit_pos - config.VALID_DAYS
    return t[~valid], j[~valid], t[valid], j[valid]


class Batcher:
    """Gathers standardised windows on the device."""

    def __init__(self, x, mean, std, device):
        self.x = torch.tensor(np.nan_to_num(np.transpose(x, (1, 0, 2))), device=device)
        self.mean = torch.tensor(mean, device=device)
        self.std = torch.tensor(std, device=device)
        self.span = torch.arange(L, device=device)
        self.device = device

    def __call__(self, t, j):
        t = torch.as_tensor(t, device=self.device)
        j = torch.as_tensor(j, device=self.device)
        xb = self.x[j[:, None], t[:, None] - L + 1 + self.span]           # (batch, L, F)
        xb = (xb - self.mean) / self.std
        r = xb[..., features.RET]
        r = r / r.std(dim=1, keepdim=True).clamp_min(1e-6)
        v = xb[..., features.VOL]
        xb = xb.clone()
        xb[..., features.RET] = r
        xb[..., features.VOL] = v - v.mean(dim=1, keepdim=True)
        return xb, torch.cumsum(r, dim=1) / L ** 0.5                     # target: cumulative path


def pooled_stats(x, t, j):
    """Feature means and deviations over the training days only. The return and
    volume channels are left unscaled here and standardised per sequence."""
    vals = x[t, j]
    mean, std = vals.mean(axis=0), vals.std(axis=0)
    mean[[features.RET, features.VOL]] = 0.0
    std[[features.RET, features.VOL]] = 1.0
    return mean.astype(np.float32), np.where(std > 0, std, 1.0).astype(np.float32)


def evaluate(model, batch, t, j, gen_seed):
    """Validation ELBO with the same sampling noise every epoch, leaving the
    training random stream untouched."""
    model.eval()
    tot, n = np.zeros(3), 0
    devices = [batch.device] if batch.device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(gen_seed)
        for s in range(0, len(t), 4 * config.BATCH):
            xb, r = batch(t[s:s + 4 * config.BATCH], j[s:s + 4 * config.BATCH])
            rec, mu, lv = model(xb)
            tot += np.array([v.item() for v in elbo_loss(rec, r, mu, lv)]) * len(xb)
            n += len(xb)
    return tot / n


def encode_all(model, batch, t, j):
    """Latent means, with dropout off."""
    model.eval()
    mus = []
    with torch.no_grad():
        for s in range(0, len(t), 4 * config.BATCH):
            mus.append(model.encode(batch(t[s:s + 4 * config.BATCH], j[s:s + 4 * config.BATCH])[0])[0].cpu())
    return torch.cat(mus).numpy()


def fit(d, x, refit_pos, device, latent, seed=config.SEED):
    """The model at the epoch of lowest validation loss, its batcher, and the
    number of training sequences, best epoch and active units."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    tt, jt, tv, jv = training_split(d, x, refit_pos)
    mean, std = pooled_stats(x, tt, jt)
    batch = Batcher(x, mean, std, device)
    model = LSTMVAE(x.shape[-1], latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.LR)

    best, best_epoch, best_state, wait = np.inf, 0, None, 0
    start = time.time()
    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(tt))
        run, seen = 0.0, 0
        for s in range(0, len(order), config.BATCH):
            idx = order[s:s + config.BATCH]
            xb, r = batch(tt[idx], jt[idx])
            opt.zero_grad()
            rec, mu, lv = model(xb)
            loss = elbo_loss(rec, r, mu, lv)[0]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.CLIP)
            opt.step()
            run += loss.item() * len(idx)
            seen += len(idx)
        v, v_rec, v_kl = evaluate(model, batch, tv, jv, seed)
        print(f"  epoch {epoch:3d}  train {run / seen:9.3f}  valid {v:9.3f}  (recon {v_rec:8.3f}, kl {v_kl:6.3f})"
              f"  {time.time() - start:7.0f}s", flush=True)
        if v < best - 1e-6:
            best, best_epoch, wait = v, epoch, 0
            best_state = {k: w.detach().clone() for k, w in model.state_dict().items()}
        else:
            wait += 1
            if wait >= config.PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("validation loss was never finite; training diverged")
    model.load_state_dict(best_state)
    act = encode_all(model, batch, tt, jt).var(axis=0)
    return model, batch, {"n_train": int(len(tt)), "best_epoch": best_epoch,
                          "active_units": int((act > config.ACTIVE_THRESHOLD).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=config.LATENT, help="latent size d")
    args = ap.parse_args()
    name = "vae" if args.latent == config.LATENT else f"vae{args.latent}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}" + (f", {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    d = data.load()
    x = features.build(d)
    refits = data.refit_dates(d["cal"])
    positions = [d["cal"].get_loc(r) for r in refits] + [len(d["cal"])]
    CACHE.mkdir(exist_ok=True)
    for k, r in enumerate(refits):
        pos, nxt = positions[k], positions[k + 1]
        print(f"refit {r.date()}, d = {args.latent}")
        model, batch, info = fit(d, x, pos, device, args.latent)
        t, j = window_days(d, x, 0, nxt)          # forecast estimated before pos, applied from pos to nxt
        torch.save(model.state_dict(), CACHE / f"{name}_{r.year}.pt")
        with open(CACHE / f"{name}_{r.year}.json", "w") as fh:
            json.dump(info, fh, indent=1)
        np.savez_compressed(CACHE / f"latent_{name}_{r.year}.npz", day=t, stock=j, mu=encode_all(model, batch, t, j))
        print(f"  active units {info['active_units']} of {args.latent}, best epoch {info['best_epoch']}, "
              f"n_train {info['n_train']}, latents saved {len(t)}")


if __name__ == "__main__":
    main()
