"""Overnight-vs-intraday return decomposition (Lou-Polk-Skouras 2019) — the published-anomaly battery.

Lou, Polk & Skouras (2019, JFE) and others (Cliff-Cooper-Gulen; Kelly et al.) document that (nearly) all of
the equity premium is earned OVERNIGHT (close→open) while the intraday (open→close) component is flat-to-
negative — a striking, persistent stylized fact. We test the tradeable long-overnight / short-intraday SPREAD
(daily return = overnight − intraday), which isolates the differential and is the honest net-of-cost test,
on SPY (its equity home) and BTC (the 24/7 crypto session, where the open/close split is a convention). The
catch is COST: capturing the overnight leg means a round-trip every day (buy at close, sell at open, and the
reverse for the intraday short), so the spread pays two flips per day — the term that kills most versions of
this trade. `overnight` is the published direction; `overnight_inverse` the mirror.

The leakage surface is ALIGNMENT: the overnight return must pair open[t] with close[t-1] (not close[t]); a
one-bar slip fabricates the whole effect. That is pinned in test_overnight.py. The strategy itself carries no
fitted signal (a fixed decomposition), so there is nothing to overfit — the guard is correctness of the
open/close pairing and the cost. PRE-REGISTERED EXPECTATION: the overnight premium is real GROSS but the
long-overnight/short-intraday spread is destroyed by the two-flips-per-day cost net of realistic fees.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from trainer import data_catalog
from trainer.summary import _capture_stats, _finite, _max_drawdown_pct, _oos_stats, _provenance_fingerprint
from trainer.walk_forward import resolve_walk_forward_window

UNIVERSES = {"spy": ["SPY"], "btc": ["BTCUSDT"]}
DEFAULT_UNIVERSE = "spy"

SIGNALS = ("overnight", "overnight_inverse")
DEFAULT_SIGNAL = "overnight"


def _load_ohlc(symbols, pairs):
    """Daily OHLC bars (price_open + close) per symbol over the requested (year, month) pairs — the raw files
    carry price_open, which the close-only cross-sectional loader discards. A symbol with nothing on disk is
    dropped rather than silently zero."""
    frames = {}
    for symbol in symbols:
        inst = data_catalog.instrument(symbol)
        directory = inst.directory if inst else "binance"
        paths = [f"{directory}/{symbol}-1d-{y}-{m}.json" for (y, m) in pairs]
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            continue
        df = pd.concat([pd.read_json(p) for p in paths], ignore_index=True)
        if not {"timestamp_close", "price_open", "price"} <= set(df.columns):
            continue
        frames[symbol] = df[["timestamp_close", "price_open", "price"]]
    return frames


def _aligned(frames):
    """symbol -> (open, close) Series on one date clock, sorted, de-duplicated."""
    out = {}
    for sym, df in frames.items():
        idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp_close"]).to_numpy())
        o = pd.Series(pd.to_numeric(df["price_open"], errors="coerce").to_numpy(), index=idx)
        c = pd.Series(pd.to_numeric(df["price"], errors="coerce").to_numpy(), index=idx)
        keep = ~idx.duplicated(keep="last")
        out[sym] = (o[keep].sort_index(), c[keep].sort_index())
    return out


def overnight_returns(frames):
    """Per-symbol overnight return: open[t] / close[t-1] − 1 (holding from the prior close to the open)."""
    out = {}
    for sym, (o, c) in _aligned(frames).items():
        out[sym] = (o / c.shift(1) - 1.0)
    return pd.DataFrame(out).sort_index()


def intraday_returns(frames):
    """Per-symbol intraday return: close[t] / open[t] − 1 (holding from the open to the close)."""
    out = {}
    for sym, (o, c) in _aligned(frames).items():
        out[sym] = (c / o - 1.0)
    return pd.DataFrame(out).sort_index()


def spread_returns(frames, signal=DEFAULT_SIGNAL):
    """Daily long-overnight / short-intraday spread return, averaged across the universe: (overnight −
    intraday), negated under `overnight_inverse`. Unknown signals are errors, not fallbacks."""
    if signal not in SIGNALS:
        raise ValueError(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    on = overnight_returns(frames)
    intra = intraday_returns(frames)
    spread = (on - intra).mean(axis=1).dropna()
    return -spread if signal == "overnight_inverse" else spread


def spread_equity(frames, signal=DEFAULT_SIGNAL, fee=0.0005, start=None):
    """Equity curve of the spread, charging `fee` on the TWO daily flips (long overnight then short intraday
    means flipping the book at the open and at the close: ~2 units of turnover per bar)."""
    r = spread_returns(frames, signal)
    if start is not None:
        r = r[r.index >= start]
    net = r - 2.0 * fee  # two flips per day
    return (1.0 + net).cumprod()


def run(cfg):
    """Run one overnight-spread cell and return a trainer-contract RunSummary."""
    universe_id = str(cfg.get("universe", DEFAULT_UNIVERSE))
    symbols = UNIVERSES.get(universe_id)
    if not symbols:
        raise SystemExit(f"unknown universe {universe_id!r}; choose one of {sorted(UNIVERSES)}")
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    fee = float(cfg.get("transaction_fee", 0.0005))

    train_pairs, test_pairs, window = resolve_walk_forward_window(cfg)
    frames = _load_ohlc(symbols, list(train_pairs) + list(test_pairs))
    if not frames:
        raise SystemExit(f"overnight needs >=1 symbol with OHLC on disk; found {len(frames)}")
    test_start = pd.to_datetime(f"{test_pairs[0][0]}-{test_pairs[0][1]:02d}-01")
    equity = spread_equity(frames, signal, fee, start=test_start)
    gross = spread_equity(frames, signal, 0.0, start=test_start)
    total_return = float(equity.iloc[-1] - 1.0) * 100 if len(equity) else 0.0
    gross_return = float(gross.iloc[-1] - 1.0) * 100 if len(gross) else 0.0
    n = max(0, len(equity) - 1)
    metrics = {
        "total_return_pct": _finite(total_return),
        "gross_return_pct": _finite(gross_return),
        "n_trades": int(2 * n),
        "realized_cost_bps": _finite(2 * n * fee * 10000),
        "universe_size": len(frames),
        "bars": int(len(equity)),
    }
    metrics.update(_oos_stats(list(equity.values)))
    metrics.update(_max_drawdown_pct(list(equity.values)))
    summary = {
        "objective": _finite(total_return),
        "metrics": metrics,
        "dataset": {
            "asset": f"{universe_id}({len(frames)})",
            "timeframe": "1d",
            "candles": int(len(equity)),
            "from": str(equity.index[0]) if len(equity) else "",
            "to": str(equity.index[-1]) if len(equity) else "",
        },
        "health": {"status": "ok", "flags": []},
        "config": dict(cfg),
        "walk_forward_window": window.get("walk_forward_window"),
    }
    try:
        summary["provenance"] = _provenance_fingerprint(cfg, dict(cfg))
    except (Exception, SystemExit):
        pass
    return summary


def main():
    parser = argparse.ArgumentParser(description="Overnight-vs-intraday spread (Lou-Polk-Skouras 2019) — published-anomaly battery")
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    with open(args.config_json) as fh:
        cfg = json.load(fh)
    summary = run(cfg)
    with open(args.summary_out, "w") as fh:
        json.dump(summary, fh)
    print(f"objective(total_return_pct)={summary['objective']:.4f} status=ok -> {args.summary_out}")


if __name__ == "__main__":
    main()
