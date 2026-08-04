"""Global context channels: mined point-in-time macro series fused onto an asset's bar clock as
RAW-LEVEL observation columns, so the model derives its own features (no hand-engineered surprises).

Representation rule (per series): a series that is ALREADY a rate / percentage / bounded ratio is shown
as its LEVEL (a static affine rescale into range — no fitted statistic, so it cannot leak); a series that
is a value / count / price / index is shown as CHANGE over time (release-over-release percent change).
Both are CAUSAL: the value at bar t depends only on releases at or before t. The provider pipeline has no
fitted/walk-forward scaler, so a full-sample standardizer would import the future — hence static-affine or
release-relative only.

Context channels are GLOBAL (asset-agnostic) and enter as per-bar COLUMNS at the provider's process_df
seam (NOT a fidelity layer, NOT env extras). They are an obs-shaping input: a context-trained checkpoint
replays only on assets exposing the SAME panel, so the panel id is part of the obs-signature. Pure (no
torch / no filesystem); the provider loads each series' observations from disk and calls
``fuse_context_series`` here. The low-level leakage-guard join lives in ``trainer.pit_fusion``.
"""

import datetime
import glob
import json
import os

from trainer.pit_fusion import (
    DEFAULT_PUBLISH_TIME_ET,
    DEFAULT_TZ,
    fuse_series,
    publish_time_for,
)

# Where each source's mined point-in-time series live on disk (relative to the run root), as written by
# the data mine: macro/{SERIES}.json (FRED), fundamentals/{TICKER_CONCEPT}.json (EDGAR).
_SOURCE_DIRS = {"fred": "macro", "edgar": "fundamentals"}

# Per-series representation + (for the LEVEL case) a static affine scale bringing the typical range into
# ~[-1, 1] with zero fitted statistics. CHANGE series need no scale (a percent is already bounded). The
# classification follows the rule above: rates/spreads = level; counts/indices/dollar values = change.
CONTEXT_SERIES = {
    "DFF": {"representation": "level", "scale": 0.1},        # fed funds effective %, ~0-8
    "DFEDTARU": {"representation": "level", "scale": 0.1},   # fed funds target upper %
    "DGS10": {"representation": "level", "scale": 0.1},      # 10y yield %
    "DFII10": {"representation": "level", "scale": 0.1},     # 10y real yield %
    "T10Y2Y": {"representation": "level", "scale": 0.3},     # 10y-2y spread (pp), ~-3..3
    "UNRATE": {"representation": "level", "scale": 0.05},    # unemployment %, ~3-15
    "CPIAUCNS": {"representation": "change"},                # CPI index (NSA)
    "CPIAUCSL": {"representation": "change"},                # CPI index (SA)
    "PCEPILFE": {"representation": "change"},                # core PCE price index
    "PAYEMS": {"representation": "change"},                  # nonfarm payrolls (count)
    "ICSA": {"representation": "change"},                    # initial jobless claims (count)
    "RSAFS": {"representation": "change"},                   # retail sales ($)
    "GDPC1": {"representation": "change"},                   # real GDP ($)
    # Price-source series: a linked/peer asset's own price, fused as its return (a price is unbounded ->
    # always CHANGE, never level). The "release" is each daily bar's close, stamped at 23:59 UTC for crypto
    # (a crypto daily bar for date D is knowable only at D's end). Used to feed one asset's context into
    # another (e.g. the majors' returns as context for an alt) — the concrete "tie assets to related assets".
    "BTCUSDT": {"source": "price", "representation": "change", "publish_time": "23:59", "tz": "UTC"},
    "ETHUSDT": {"source": "price", "representation": "change", "publish_time": "23:59", "tz": "UTC"},
    # Keyless macro-risk proxies (yfinance daily; mined into commodities/ + etfs/). Stamped at the US
    # equity close — a crypto daily bar (23:59 UTC ≈ 19:00 ET) sees the proxy's same-day close, leakage-safe.
    "GOLD": {"source": "price", "representation": "change", "publish_time": "16:00", "tz": "America/New_York"},
    "SPY": {"source": "price", "representation": "change", "publish_time": "16:00", "tz": "America/New_York"},
    "UUP": {"source": "price", "representation": "change", "publish_time": "16:00", "tz": "America/New_York"},
}

# Named reusable panels — the `context_set` lever picks one. A context-trained model's checkpoint replays
# only on assets exposing the SAME panel, so a fixed shared panel (not per-asset edges) is what makes
# cross-asset replay possible; the panel id is stamped into the obs-signature.
CONTEXT_PANELS = {
    "none": [],
    "rates": ["DFF", "DGS10", "DFII10", "T10Y2Y"],
    "macro_core": ["DFF", "DGS10", "T10Y2Y", "UNRATE", "CPIAUCNS", "PAYEMS"],
    # Major-crypto returns as context (use for an ALT — degenerate if the traded asset is itself a major).
    # Price-source, so it works on data already on disk (no external mine).
    "majors": ["BTCUSDT", "ETHUSDT"],
    # Keyless macro-risk panel: gold + broad equities (SPY) + the dollar (UUP), as daily returns. yfinance,
    # no FRED key — the risk-appetite series crypto co-moves with.
    "market": ["GOLD", "SPY", "UUP"],
}
DEFAULT_CONTEXT = "none"


def context_panel_ids():
    """Selectable context panels, `none` first."""
    return list(CONTEXT_PANELS.keys())


def resolve_context(cfg=None):
    """Return ``(panel_id, [series_spec])`` for the run's ``context_set`` lever.

    Each spec = {id, source, representation, scale, publish_time, tz} — everything the provider needs to
    load the series and fuse it leakage-safely. Raises ``SystemExit`` on an unknown panel — fail fast.
    """
    cfg = cfg or {}
    raw = cfg.get("context_set")
    panel = str(raw) if raw not in (None, "") else DEFAULT_CONTEXT
    if panel not in CONTEXT_PANELS:
        raise SystemExit(f"unknown context_set {panel!r} — choices: {context_panel_ids()}")
    return panel, [_series_spec(sid) for sid in CONTEXT_PANELS[panel]]


def _series_spec(series_id):
    meta = CONTEXT_SERIES[series_id]
    return {
        "id": series_id,
        "source": meta.get("source", "fred"),
        "representation": meta["representation"],
        "scale": meta.get("scale", 1.0),
        "publish_time": meta.get("publish_time") or publish_time_for(series_id),
        "tz": meta.get("tz", DEFAULT_TZ),
    }


def fuse_context_series(bar_ms, observations, representation="level", scale=1.0,
                        publish_time=DEFAULT_PUBLISH_TIME_ET, tz=DEFAULT_TZ):
    """Fuse a point-in-time ``observations`` series onto ``bar_ms`` as a represented raw column (a list,
    one value per bar). ``level`` = the forward-filled raw level x ``scale`` (None before the first
    release). ``change`` = the release-over-release percent change, stamped at the LATER release so it is
    knowable only from that bar (None before the second release). Causal in both cases."""
    if representation == "change":
        observations = _release_over_release_change(observations)
    fused = fuse_series(bar_ms, observations, publish_time, tz)
    if scale == 1.0:
        return fused
    return [None if v is None else v * scale for v in fused]


def _release_over_release_change(observations):
    """Turn a level series into a percent-change series stamped at the LATER release's date (the change
    from the prior value is knowable only when the new value is released). The first release yields no
    change; a zero prior is skipped (undefined percent change)."""
    ordered = sorted(
        (o for o in observations if o.get("releaseDate") is not None),
        key=lambda o: o["releaseDate"],
    )
    out = []
    prev = None
    for obs in ordered:
        value = obs["value"]
        if prev not in (None, 0):
            out.append({"releaseDate": obs["releaseDate"], "value": (value - prev) / prev})
        prev = value
    return out


def context_columns(timestamps, series_specs, load_observations):
    """Build ``{column_name: fused_values}`` for each resolved context series. ``load_observations(id,
    source)`` returns that series' point-in-time observations. Column name = ``context_<id>``. Pure — the
    disk read is injected so this is unit-testable without a provider."""
    columns = {}
    for spec in series_specs:
        obs = load_observations(spec["id"], spec.get("source", "fred"))
        columns[f"context_{spec['id']}"] = fuse_context_series(
            timestamps,
            obs,
            representation=spec["representation"],
            scale=spec.get("scale", 1.0),
            publish_time=spec["publish_time"],
            tz=spec["tz"],
        )
    return columns


def load_context_observations(series_id, source="fred", root="."):
    """Read a mined RELEASE series' point-in-time observations (``[{releaseDate, value, ...}]``) from disk.
    Fails fast when the file is absent — a context run must never silently train on a zeroed channel; the
    data must be mined first."""
    directory = _SOURCE_DIRS.get(source, source)
    path = os.path.join(root, directory, f"{series_id}.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"context series {series_id!r} not on disk at {path} — mine it first "
            f"(the data mine writes {directory}/{series_id}.json)."
        )
    with open(path) as handle:
        return json.load(handle)


def _price_directory(symbol):
    """The on-disk directory for a price-source series, resolved from the data catalog (crypto -> binance,
    GOLD -> commodities, SPY/UUP -> etfs); defaults to binance for an uncatalogued symbol."""
    from trainer import data_catalog

    inst = data_catalog.instrument(symbol)
    return inst.directory if inst else "binance"


def load_price_observations(symbol, root="."):
    """Read a PRICE-source series (a linked/peer/market asset's daily klines) into point-in-time
    observations: each daily bar's close, stamped at the bar's close DATE (the release; publish_time/tz on
    the series spec place it at the actual close instant). Value = the close; the return is derived
    downstream by the 'change' representation. The directory is resolved from the catalog so crypto peers
    and market proxies share this loader. Fails fast when no klines are on disk."""
    directory = _price_directory(symbol)
    paths = sorted(glob.glob(os.path.join(root, directory, f"{symbol}-1d-*.json")))
    if not paths:
        raise SystemExit(
            f"price-source context {symbol!r} not on disk ({directory}/{symbol}-1d-*.json) — needs daily klines."
        )
    observations = []
    for path in paths:
        with open(path) as handle:
            rows = json.load(handle)
        for row in rows:
            close_date = datetime.datetime.utcfromtimestamp(int(row["timestamp_close"]) / 1000).strftime("%Y-%m-%d")
            # klines store OHLCV as strings in some dumps — coerce so the return math never hits str - str.
            observations.append({"releaseDate": close_date, "value": float(row["price"])})
    return observations


def load_series_observations(series_id, source="fred", root="."):
    """Dispatch to the right loader for a context series' ``source``: price-source (a peer asset's klines)
    vs a mined release series (FRED macro / EDGAR fundamentals)."""
    if source == "price":
        return load_price_observations(series_id, root=root)
    return load_context_observations(series_id, source=source, root=root)
