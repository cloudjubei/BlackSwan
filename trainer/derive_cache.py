"""Derive higher-fidelity klines from the canonical 1m source + cache them to disc.

1m is the single source of truth; 1h/1d bars are DERIVED from it (never separately mined), and the
derived bars are cached under ``binance/derived/`` so a run reads them directly instead of re-reading
~1.5GB of 1m JSON and re-aggregating every time. The cache is rebuilt for a month whenever its 1m
source is newer (mtime check). Derived bars carry no ``indicators`` dict — indicators are a data-mine
artifact computed separately, not a reason to adopt separately-mined native-fidelity files.
"""

import os

import pandas as pd

_CACHE_DIR = "binance/derived"
# Output kline columns, matching the raw kline JSON shape the data providers read.
_OHLCV_SUM = ["volume", "asset_volume_quote", "trades_number", "asset_volume_taker_base", "asset_volume_taker_quote"]


def _to_ms(series):
    """Coerce a timestamp column (datetime64 after pd.read_json, or raw ms ints) to integer ms."""
    s = pd.Series(series)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="ms").astype("int64") // 10**6
    return pd.to_datetime(s).astype("int64") // 10**6


def derive_bars(df, bucket_ms):
    """Aggregate 1m klines into fixed-width buckets (e.g. 3_600_000 for 1h) — open=first, high=max,
    low=min, close=last, the volume/quote/trade/taker columns summed. Matches the exchange's native
    higher-fidelity bar (verified: summed 1m == native 1h)."""
    if df.empty or "timestamp" not in df.columns:
        cols = ["timestamp", "price_open", "price_high", "price_low", "price", "timestamp_close", *_OHLCV_SUM]
        return pd.DataFrame({c: pd.Series([], dtype="float64") for c in cols})
    d = df.copy()
    open_ms = _to_ms(d["timestamp"])
    close_ms = _to_ms(d["timestamp_close"])
    d["_bucket"] = (open_ms // bucket_ms) * bucket_ms
    d["_close_ms"] = close_ms
    for c in ["price_open", "price_high", "price_low", "price", *_OHLCV_SUM]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    g = d.groupby("_bucket", sort=True)
    out = pd.DataFrame(
        {
            "timestamp": g["_bucket"].first().astype("int64"),
            "price_open": g["price_open"].first(),
            "price_high": g["price_high"].max(),
            "price_low": g["price_low"].min(),
            "price": g["price"].last(),
            "timestamp_close": g["_close_ms"].last().astype("int64"),
        }
    )
    for c in _OHLCV_SUM:
        if c in d.columns:
            out[c] = g[c].sum()
    return out.reset_index(drop=True)


def ensure_derived(symbol, pairs, fidelity, cache_dir=_CACHE_DIR):
    """Ensure derived `fidelity` klines for each (year, month) in `pairs` exist + are fresh vs their 1m
    source, deriving + caching any that are missing/stale. Returns the existing derived file paths."""
    bucket_ms = {"1h": 3_600_000, "1d": 86_400_000}[fidelity]
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for (year, month) in pairs:
        src = f"binance/{symbol}-1m-{year}-{month}.json"
        if not os.path.exists(src):
            continue
        dst = os.path.join(cache_dir, f"{symbol}-{fidelity}-{year}-{month}.json")
        if not (os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src)):
            derive_bars(pd.read_json(src), bucket_ms).to_json(dst, orient="records")
        out.append(dst)
    return out
