"""Intraday conditional-edge backtest (the first decision-frequency line in the project).

Every trading line so far decided ONCE per day and, across hundreds of cells, found no directional
skill — only exposure reduction. This module changes the question from "which day is up?" to "in the
minutes after a recognisable CONDITION, does a short-horizon directional trade capture a move large
enough to clear the crypto round trip (~0.2%)?". The round-trip fee is the ENEMY here: at intraday
cadence the cost floor is paid over and over, so an edge that does not clear it on a PER-TRADE basis is
worse than doing nothing. This backtest exists to MEASURE that clearance honestly, not to manufacture a
win, which is why the two lookahead traps it is most exposed to are pinned in test_intraday.py before a
number is allowed out.

The two traps, both silent and both fatal to the result:

* A position applied to bar t's return must be decided ONLY from data at/before t-1. A breakout is
  detected from the return REALISED over bar t and its trailing vol through t-1, and the resulting
  position is applied to bars t+1 .. t+hold_bars — never bar t's own return. The whole pipeline
  (1m -> resample -> signal -> position) is time-prefix causal: corrupt the price series after any cut
  and every earlier position is byte-identical.

* "Is this a high-vol bar?" must be judged against the TRAILING, past-only vol distribution. Labelling
  a bar high-vol because it sits in the top decile of the WHOLE test sample is a classic silent
  lookahead — the sample's later high-vol bars pull the threshold to a level the early bars are then
  scored against with knowledge they could not have had. regime_flags compares v_{t-1} against the
  quantile of {v_0 .. v_{t-1}} only, so a bar that is extreme versus the full sample but ordinary
  versus its own past is correctly left flat.

Deterministic and model-free (a seed lever exists for contract conformance only). It reuses summary.py's
equity-curve statistics verbatim — _oos_stats (the per-step Sharpe the DSR gate consumes, NOT annualised),
_max_drawdown_pct, _capture_stats (beta/up/down versus the asset), and the fee-charged buy-and-hold
benchmark — so every existing gate, lens and scorecard reads an intraday cell unchanged. The one metric
this line adds to the vocabulary is trades_per_day: the probe only means anything if it trades several
times a day, so the round-trip rate is emitted and pinned.
"""

import argparse
import datetime
import json
import os

import numpy as np

from trainer import data_catalog
from trainer.summary import (
    _capture_stats,
    _finite,
    _hold_equity,
    _max_drawdown_pct,
    _oos_stats,
    _provenance_fingerprint,
)
from trainer.walk_forward import resolve_walk_forward_window

# The conditions the probe can trade. breakout_momentum chases a fresh volatility-spike breakout;
# regime_momentum / regime_reversion gate on a high-vol REGIME and then trade with / against the last
# move. An unknown value is refused rather than defaulted — a typo'd signal that silently ran breakout
# would file its result under the wrong label and corrupt the evidence trail.
SIGNALS = ("breakout_momentum", "regime_momentum", "regime_reversion")
DEFAULT_SIGNAL = "breakout_momentum"

DEFAULT_ASSET = "BTCUSDT"
DEFAULT_BAR = 60  # decision-bar width in minutes (1m resampled up to this)
DEFAULT_VOL_WINDOW = 20  # bars of trailing returns behind the realized-vol estimate
DEFAULT_BREAKOUT_K = 2.0
DEFAULT_REGIME_PCT = 0.80
DEFAULT_HOLD_BARS = 4
DEFAULT_FEE = 0.001  # per side; the round trip (1-fee)^2 is ~0.2%, the crypto cost floor this probe fights

_DAY_MS = 86_400_000


def _sign(x):
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


def _is_num(x):
    return isinstance(x, (int, float)) and np.isfinite(x)


# --- 1m -> decision-bar resample ----------------------------------------------------------------------


def resample(rows, bar_minutes):
    """Aggregate contiguous 1-minute OHLCV rows into decision bars of ``bar_minutes`` minutes.

    Each bar is bucketed by ``timestamp // (bar_minutes * 60_000)`` — wall-clock aligned to the epoch —
    so for any bar width that divides the day evenly a bar never straddles a month boundary, and per-file
    resampling then concatenation equals resampling the whole span at once. A bar's OHLC is built from
    ONLY the minutes inside its own bucket: open = the bucket's first minute's open, high/low = the
    extremes over the bucket, close = the last minute's close, volume = the sum. Nothing from the next
    bucket can leak backwards. Prices arrive as strings and are cast here. Rows are assumed time-ordered,
    which the binance 1m files are within a month; an empty input yields empty columns, never a NaN.
    """
    out = {"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
    if not rows:
        return out
    width = int(bar_minutes) * 60_000

    def flush(cur):
        out["timestamp"].append(cur["ts"])
        out["open"].append(cur["o"])
        out["high"].append(cur["h"])
        out["low"].append(cur["l"])
        out["close"].append(cur["c"])
        out["volume"].append(cur["v"])

    cur = None
    for row in rows:
        ts = int(row["timestamp"])
        bucket = ts // width
        o = float(row["price_open"])
        h = float(row["price_high"])
        low = float(row["price_low"])
        c = float(row["price"])
        v = float(row.get("volume", 0.0) or 0.0)
        if cur is None or bucket != cur["bucket"]:
            if cur is not None:
                flush(cur)
            cur = {"bucket": bucket, "ts": bucket * width, "o": o, "h": h, "l": low, "c": c, "v": v}
        else:
            cur["h"] = max(cur["h"], h)
            cur["l"] = min(cur["l"], low)
            cur["c"] = c
            cur["v"] += v
    if cur is not None:
        flush(cur)
    return out


# --- returns, trailing vol, and the regime gate -------------------------------------------------------


def bar_returns(closes):
    """Per-bar simple return, each measured against the bar's OWN previous close. ``None`` for the first
    bar (no prior) and wherever a close is non-positive, so an undefined return can never masquerade as 0."""
    out = [None]
    for i in range(1, len(closes)):
        prev, cur = closes[i - 1], closes[i]
        if _is_num(prev) and _is_num(cur) and prev > 0:
            out.append(cur / prev - 1.0)
        else:
            out.append(None)
    return out


def realized_vol(returns, vol_window):
    """Trailing realized volatility: v[t] = std of the ``vol_window`` returns ending at t, using only
    past-and-present data. ``None`` until a full window of defined returns is available (returns begin at
    index 1, so v[t] is undefined for t < vol_window and wherever the window contains a gap)."""
    w = int(vol_window)
    n = len(returns)
    out = [None] * n
    if w < 1:
        return out
    for t in range(w, n):
        window = returns[t - w + 1 : t + 1]
        if all(_is_num(r) for r in window):
            out[t] = float(np.std(window))
    return out


def regime_flags(vol, regime_pct, min_history=2):
    """Per-bar high-vol REGIME flag, judged strictly against the TRAILING vol distribution.

    flag[t] is True when v[t-1] sits at or above the ``regime_pct`` quantile of the vol values observed
    up to and including t-1 — i.e. the distribution {v_0 .. v_{t-1}} known at the moment the decision for
    bar t+1 is made. This is the one place a whole-sample quantile would be a silent lookahead: scoring an
    early bar against a threshold set by the sample's later high-vol episodes hands it knowledge of the
    future and fabricates regime hits. A bar that is extreme versus the full sample but ordinary versus
    its own past is therefore left flat. ``min_history`` finite observations are required before any bar
    can be flagged, so a thin warm-up distribution never produces a spurious regime.
    """
    n = len(vol)
    flags = [False] * n
    history = []  # finite vol values observed so far, in time order (indices 0 .. t-1)
    for t in range(n):
        prev = vol[t - 1] if t - 1 >= 0 else None
        if _is_num(prev) and len(history) >= int(min_history):
            threshold = float(np.quantile(history, float(regime_pct)))
            flags[t] = bool(prev >= threshold)
        if _is_num(vol[t]):
            history.append(float(vol[t]))
    return flags


# --- signal -> positions ------------------------------------------------------------------------------


def _build_positions(n, side_at, hold_bars, start_index):
    """Turn a per-bar entry decision into a position vector under the shared hold-then-flat mechanic.

    ``side_at(t)`` returns the side (+1/-1) to enter when a fresh signal fires at bar t while flat, or
    ``None``/0 for no entry. On a fire the position occupies bars t+1 .. t+hold_bars (it earns those
    returns, never bar t's own), then scanning resumes at t+hold_bars+1 — so a new signal that fires
    inside the hold window is ignored, and no two round-trips are ever adjacent. Entries before
    ``start_index`` are suppressed (the formation lookback warms vol without trading), which keeps every
    accounted trade inside the test span. pos[t] stays 0 wherever undefined, so the book is always finite.
    """
    pos = np.zeros(n)
    hold = max(1, int(hold_bars))
    t = 0
    while t < n:
        side = side_at(t) if t >= start_index else None
        if side:
            end = min(t + hold, n - 1)
            pos[t + 1 : end + 1] = side
            t = end + 1
        else:
            t += 1
    return pos


def _breakout_side(returns, vol, k):
    """A breakout at t fires when |return_t| exceeds k * trailing vol through t-1; the entry side is the
    sign of that return (momentum). Undefined vol or return -> no fire."""
    kk = float(k)

    def side_at(t):
        r = returns[t] if 0 <= t < len(returns) else None
        v_prev = vol[t - 1] if t - 1 >= 0 else None
        if _is_num(r) and _is_num(v_prev) and abs(r) > kk * v_prev:
            return _sign(r)
        return None

    return side_at


def _regime_side(returns, flags, reversion):
    """While in a high-vol regime and flat, enter the sign of the last bar's return (momentum) or its
    negation (reversion). Undefined return -> no fire."""

    def side_at(t):
        if not flags[t]:
            return None
        r = returns[t] if 0 <= t < len(returns) else None
        if not _is_num(r):
            return None
        s = _sign(r)
        return (-s if reversion else s) or None

    return side_at


def positions_from_closes(closes, cfg, start_index=0):
    """Decision-bar position vector for the configured signal, over a series of decision-bar closes.

    The returns and trailing vol are computed over the WHOLE series (so the formation lookback can warm
    them), but entries are gated to ``start_index`` onward. Dispatches on ``cfg['signal']``; an unknown
    signal is refused rather than silently treated as a breakout."""
    n = len(closes)
    if n == 0:
        return np.zeros(0)
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    vol_window = int(cfg.get("vol_window", DEFAULT_VOL_WINDOW))
    hold_bars = int(cfg.get("hold_bars", DEFAULT_HOLD_BARS))
    returns = bar_returns(closes)
    vol = realized_vol(returns, vol_window)
    if signal == "breakout_momentum":
        side_at = _breakout_side(returns, vol, float(cfg.get("breakout_k", DEFAULT_BREAKOUT_K)))
    else:
        flags = regime_flags(vol, float(cfg.get("regime_pct", DEFAULT_REGIME_PCT)))
        side_at = _regime_side(returns, flags, reversion=(signal == "regime_reversion"))
    return _build_positions(n, side_at, hold_bars, start_index)


def positions_from_rows(rows, cfg, start_index=0):
    """End-to-end from raw 1m rows: resample to decision bars, then build the position vector. Returns
    ``(timestamps, positions)`` so a caller (and the causality test) can align positions to bar time."""
    bars = resample(rows, int(cfg.get("bar", DEFAULT_BAR)))
    return bars["timestamp"], positions_from_closes(bars["close"], cfg, start_index)


# --- backtest, costs, and trade counting --------------------------------------------------------------


def backtest(closes, pos, fee):
    """Equity curve of the position book, starting at 1.0. Each step earns pos[i] * return_i and pays
    ``fee`` on the turnover |pos[i] - pos[i-1]| — so a round trip is charged once at entry and once at
    exit. A flat or one-bar series simply stays at 1.0; everything is finite."""
    n = len(closes)
    if n == 0:
        return []
    returns = bar_returns(closes)
    fee = float(fee)
    equity = [1.0]
    for i in range(1, n):
        r = returns[i]
        gross = _finite(pos[i] * r) if _is_num(r) else 0.0
        turnover = abs(_finite(pos[i]) - _finite(pos[i - 1]))
        equity.append(_finite(equity[-1] * (1.0 + gross) * (1.0 - fee * turnover), equity[-1]))
    return equity


def round_trips(pos):
    """Number of round-trips: each contiguous run of non-zero position is one entry+exit. Counted as the
    0 -> non-zero transitions (runs are always separated by a flat bar, so this equals the entry count)."""
    n = 0
    for i in range(len(pos)):
        if pos[i] != 0.0 and (i == 0 or pos[i - 1] == 0.0):
            n += 1
    return n


def trades_per_day(n_round_trips, first_ts, last_ts):
    """Round-trips per calendar DAY over the accounted test span (its first to last bar timestamp). 0.0
    for a zero-length span, never a divide-by-zero. This is the probe's headline liveness metric: the
    thesis is only tested if the strategy trades several times a day."""
    days = (int(last_ts) - int(first_ts)) / _DAY_MS
    return _finite(n_round_trips / days) if days > 0 else 0.0


def _trade_edges(closes, pos, fee):
    """Per-trade NET return (percent): for each round-trip, the compounded pos*return over its held bars,
    charged the round-trip fee both ways. The mean of these is the honest per-trade edge — positive only
    when the captured move actually clears the cost floor."""
    returns = bar_returns(closes)
    fee = float(fee)
    edges = []
    i = 1
    n = len(closes)
    while i < n:
        if pos[i] != 0.0 and pos[i - 1] == 0.0:
            growth = 1.0
            j = i
            while j < n and pos[j] != 0.0:
                r = returns[j]
                if _is_num(r):
                    growth *= 1.0 + pos[j] * r
                j += 1
            edges.append((growth * (1.0 - fee) ** 2 - 1.0) * 100.0)
            i = j
        else:
            i += 1
    return edges


# --- data loading -------------------------------------------------------------------------------------


def _read_rows(path):
    with open(path) as fh:
        return json.load(fh)


def _load_bars(asset, pairs, bar_minutes):
    """Decision bars for ``asset`` over the requested (year, month) pairs, resampled per file and
    concatenated. Missing months are skipped rather than faked. Resampling per file keeps memory bounded
    (bar arrays, not the raw 1m rows) and is exact for day-dividing bar widths."""
    inst = data_catalog.instrument(asset)
    directory = inst.directory if inst else "binance"
    out = {"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
    for (y, m) in pairs:
        path = f"{directory}/{asset}-1m-{y}-{m}.json"
        if not os.path.exists(path):
            continue
        bars = resample(_read_rows(path), bar_minutes)
        for key in out:
            out[key].extend(bars[key])
    return out


def _month_start_ms(ym):
    y, m = (int(x) for x in str(ym).split("-"))
    return int(datetime.datetime(y, m, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000)


def _prior_month(pair):
    y, m = int(pair[0]), int(pair[1])
    return (y - 1, 12) if m == 1 else (y, m - 1)


def _iso_from_ms(value):
    try:
        return datetime.datetime.fromtimestamp(float(value) / 1000.0, datetime.timezone.utc).isoformat()
    except Exception:
        return None


# --- the run contract ---------------------------------------------------------------------------------


def run(cfg):
    """Run one intraday cell and return a trainer-contract RunSummary.

    A short formation lookback (the month before the test window) warms the trailing vol and regime
    distribution; only the TEST span is accounted, exactly as the daily lines report. The objective is
    the per-step OOS Sharpe; return_vs_hold_pct is measured against a fee-charged buy-and-hold of the
    same asset over the same span.
    """
    asset = str(cfg.get("asset", DEFAULT_ASSET))
    bar_minutes = int(cfg.get("bar", DEFAULT_BAR))
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    fee = float(cfg.get("transaction_fee", DEFAULT_FEE))

    _, test_pairs, meta = resolve_walk_forward_window(cfg)
    load_pairs = [_prior_month(test_pairs[0])] + list(test_pairs)
    bars = _load_bars(asset, load_pairs, bar_minutes)
    closes = bars["close"]
    timestamps = bars["timestamp"]

    test_from_ms = _month_start_ms(meta["test_from"])
    start_index = next((i for i, ts in enumerate(timestamps) if ts >= test_from_ms), len(timestamps))

    pos = positions_from_closes(closes, cfg, start_index=start_index)
    closes_t = closes[start_index:]
    pos_t = pos[start_index:]
    ts_t = timestamps[start_index:]

    equity = backtest(closes_t, pos_t, fee)
    n_rt = round_trips(pos_t)
    total_return_pct = (equity[-1] - 1.0) * 100.0 if len(equity) >= 1 else 0.0
    turnover = float(np.abs(np.diff(np.concatenate([[0.0], np.asarray(pos_t, dtype=float)]))).sum())

    metrics = {
        "total_return_pct": _finite(total_return_pct),
        "baseline": 0.0,
        "n_trades": n_rt,
        "realized_cost_bps": _finite(turnover * fee * 10000),
        "final_net_worth": _finite(equity[-1]) if equity else 1.0,
        "trades_per_day": trades_per_day(n_rt, ts_t[0], ts_t[-1]) if len(ts_t) >= 2 else 0.0,
    }
    metrics.update(_oos_stats(equity))
    metrics.update(_max_drawdown_pct(equity))
    metrics.update(_capture_stats(equity, closes_t))

    edges = _trade_edges(closes_t, pos_t, fee)
    if edges:
        metrics["signal_expectancy"] = _finite(sum(edges) / len(edges))
        metrics["signal_hit_rate"] = _finite(100.0 * sum(1 for e in edges if e > 0) / len(edges))
        metrics["signal_count"] = len(edges)
    else:
        metrics["signal_expectancy"] = 0.0

    benchmark = {}
    prices = [p for p in closes_t if _is_num(p) and p > 0]
    if len(prices) >= 2:
        round_trip = (1.0 - fee) ** 2
        hold_return_pct = (prices[-1] / prices[0] * round_trip - 1.0) * 100.0
        benchmark["hold_return_pct"] = _finite(hold_return_pct)
        hold_equity = _hold_equity(prices, round_trip)
        hstats = _oos_stats(hold_equity)
        if "oos_sharpe" in hstats:
            benchmark["hold_sharpe"] = hstats["oos_sharpe"]
        hdd = _max_drawdown_pct(hold_equity)
        if "max_drawdown_pct" in hdd:
            benchmark["hold_max_drawdown_pct"] = hdd["max_drawdown_pct"]
        metrics["hold_return_pct"] = benchmark["hold_return_pct"]
        metrics["return_vs_hold_pct"] = _finite(total_return_pct - benchmark["hold_return_pct"])
        metrics["hold_net_of_fees"] = True

    objective = metrics.get("oos_sharpe", 0.0)
    stored_cfg = dict(cfg)
    summary = {
        "objective": _finite(objective),
        "metrics": metrics,
        "health": {"status": "ok", "flags": []},
        "config": stored_cfg,
        "dataset": {
            "asset": asset,
            "timeframe": f"{bar_minutes}m",
            "candles": int(len(closes_t)),
            "walk_forward_window": meta["walk_forward_window"],
            "from": _iso_from_ms(ts_t[0]) if ts_t else None,
            "to": _iso_from_ms(ts_t[-1]) if ts_t else None,
        },
        "walk_forward_window": meta["walk_forward_window"],
    }
    if benchmark:
        summary["benchmark"] = benchmark
    try:
        summary["provenance"] = {"ranAt": cfg.get("ran_at"), **_provenance_fingerprint(cfg, stored_cfg)}
    except Exception:
        summary["provenance"] = {}
    if "seed" in cfg:
        summary["seed"] = int(cfg["seed"])
    return summary


def main():
    parser = argparse.ArgumentParser(description="Intraday conditional-edge backtest")
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    with open(args.config_json) as fh:
        cfg = json.load(fh)
    summary = run(cfg)
    with open(args.summary_out, "w") as fh:
        json.dump(summary, fh)
    m = summary["metrics"]
    print(
        f"objective(oos_sharpe)={summary['objective']:.4f} "
        f"total_return_pct={m.get('total_return_pct', 0.0):.4f} "
        f"trades_per_day={m.get('trades_per_day', 0.0):.4f} "
        f"return_vs_hold_pct={m.get('return_vs_hold_pct', 0.0):.4f} -> {args.summary_out}"
    )


if __name__ == "__main__":
    main()
