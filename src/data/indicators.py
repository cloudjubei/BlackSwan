"""Technical indicators computed at RUNTIME from raw OHLCV.

The stored data is kept minimal (1m OHLCV is the source of truth; higher fidelities are derived — see
trainer/derive_cache.py). Indicators are derived here in code, not baked into the data, so the data stays
small + cheap to transfer and any indicator fix is a one-line change rather than re-deriving gigabytes.

Formulas + normalisations mirror the BlackSwanPriceEmitter so the values match its precomputed
`indicators` dict (verified to ~1e-6). Each function takes pandas Series + a period and returns a
per-bar Series — the indicator over the trailing `period`-window ending at each bar (NaN until the
window fills; the callers fill NaN with 0, matching the emitter's "insufficient history -> 0").
"""

import numpy as np
import pandas as pd


def rsi(price, period):
    """RSI over simple (not Wilder) averages, normalised to [-1,1] via rsi/50 - 1."""
    delta = price.diff()
    avg_gain = delta.clip(lower=0).rolling(period).mean()
    avg_loss = (-delta).clip(lower=0).rolling(period).mean()
    rs = np.where(avg_loss > 0, avg_gain / avg_loss.where(avg_loss > 0), 1e7)
    return pd.Series(100 - 100 / (1 + rs), index=price.index) / 50 - 1


def williams(price, period):
    """Williams %R over close prices, mapped to [-1,1]."""
    hh = price.rolling(period).max()
    ll = price.rolling(period).min()
    return (hh - price) / (hh - ll) * 2 - 1


def stochastic(price, low, high, period):
    """Stochastic %K over the high/low range, mapped to [-1,1]."""
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return (price - ll) / (hh - ll) * 2 - 1


def choppiness(price, low, high, period):
    """Choppiness index over true-range / true-range-of-range, mapped to [-1,1]."""
    prev_close = price.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    true_range = tr.rolling(period).sum()
    true_high = pd.concat([high, prev_close], axis=1).max(axis=1).rolling(period).max()
    true_low = pd.concat([low, prev_close], axis=1).min(axis=1).rolling(period).min()
    idx = np.log10(true_range / (true_high - true_low)) / np.log10(period)
    return pd.Series(idx, index=price.index) * 2 - 1


def mean_reversion(price, period):
    """z-score of price vs its rolling mean (population sd). Unbounded — squash before use."""
    sma = price.rolling(period).mean()
    sd = price.rolling(period).std(ddof=0)
    return (price - sma) / sd


def turbulence(price, period):
    """Mahalanobis distance² of the latest return vs the window's return distribution. Unbounded."""
    ret = price.pct_change()
    mean = ret.rolling(period).mean()
    var = ret.rolling(period).var(ddof=0)
    z = (ret - mean) / np.sqrt(var)
    return z * z


def obv(price, volume, period):
    """On-balance volume over the window, normalised by total window volume to [-1,1]-ish."""
    signed = np.sign(price.diff()) * volume
    obv_window = volume.shift(period - 1) + signed.rolling(period - 1).sum()
    return obv_window / volume.rolling(period).sum() * 2 - 1
