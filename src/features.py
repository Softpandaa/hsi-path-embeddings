"""The sixteen daily input features of Table A.1, made scale free."""

import numpy as np
import pandas as pd

NAMES = ("open_gap", "high_gap", "low_gap", "ret", "log_volume", "tr", "atr20",
         "sma5", "sma10", "sma20", "ema5", "ema10", "ema20", "k14", "d14", "rsi14")
RET, VOL = NAMES.index("ret"), NAMES.index("log_volume")


def _ratio(num, den, fill):
    """num / den where den > 0, fill where the range is empty (flat prices)."""
    out = num / den.where(den > 0)
    return out.where(den.isna() | (den > 0), fill)


def build(d):
    """Array of shape (days, stocks, 16), NaN where a feature is not yet defined."""
    o, h, lo, c, a, v = d["open"], d["high"], d["low"], d["close"], d["adj"], d["volume"]
    prev = c.shift(1)
    tr = np.fmax(np.fmax((h - lo).values, (h - prev).abs().values), (lo - prev).abs().values)
    tr = pd.DataFrame(tr, index=c.index, columns=c.columns).where(prev.notna())
    lo14, hi14 = lo.rolling(14).min(), h.rolling(14).max()
    k = _ratio(c - lo14, hi14 - lo14, 0.5)
    up, dn = c.diff().clip(lower=0), (-c.diff()).clip(lower=0)
    g = up.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    s = dn.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    feats = [
        o / prev - 1, h / prev - 1, lo / prev - 1,
        np.log(a / a.shift(1)),
        np.log(v),
        tr / c, tr.rolling(20).mean() / c,
        c / c.rolling(5).mean() - 1, c / c.rolling(10).mean() - 1, c / c.rolling(20).mean() - 1,
        c / c.ewm(span=5, adjust=False, min_periods=5).mean() - 1,
        c / c.ewm(span=10, adjust=False, min_periods=10).mean() - 1,
        c / c.ewm(span=20, adjust=False, min_periods=20).mean() - 1,
        k, k.rolling(3).mean(),
        _ratio(g, g + s, 0.5),
    ]
    return np.stack([f.values.astype(np.float32) for f in feats], axis=-1)
