"""COT probe — the POSITIONING / FLOW signal class (the one class the program had not tested).

CFTC Commitments-of-Traders MANAGED-MONEY net positioning (net = spec long - short, as a fraction of open
interest) is an extreme-of-past signal with a real structural story the pure-price signals lack: when
speculators are crowded to one side they become FORCED unwinders on a reversal (a nameable counterparty). This
asks whether a crowded-positioning extreme times the metal — FADE it (cot_contrarian: the crowded side unwinds)
or RIDE it (cot_momentum). It reuses the shared past-only extreme core (signal_extremes) and the intraday
backtest mechanics, so every gate/lens/scorecard reads a COT cell unchanged. The one surface new here is the COT
JOIN, which carries the leakage trap pinned in test_cot.py: the COT measured on a Tuesday is PUBLISHED the
following Friday, so a crypto/commodity bar dated D reads only a report whose (report date + release_lag_days) is
at/before D — never a report not yet public. The extreme is past-only (delegated), the position applies to the
next bar, positions never read price. Deterministic and model-free (the seed is contract-only).

DATA: cot/<ASSET>.json = { "YYYY-MM-DD": {"net": .., "oi": ..}, ... } (report Tuesday), from scripts/fetch_cot.py.
"""

import argparse
import datetime
import json
import os

import numpy as np

from trainer.intraday import (
    _build_positions,
    _is_num,
    _iso_from_ms,
    _load_bars,
    _month_start_ms,
    _prior_month,
    _trade_edges,
    backtest,
    round_trips,
    trades_per_day,
)
from trainer.signal_extremes import past_only_extreme_sides
from trainer.summary import (
    _capture_stats,
    _finite,
    _hold_equity,
    _max_drawdown_pct,
    _oos_stats,
    _provenance_fingerprint,
)
from trainer.walk_forward import resolve_walk_forward_window

# cot_contrarian FADES a crowded positioning extreme (top-tail net-long -> short); cot_momentum RIDES it. An
# unknown value is refused so a typo can never file one arm's result under the other's label.
SIGNALS = ("cot_contrarian", "cot_momentum")
DEFAULT_SIGNAL = "cot_contrarian"

DEFAULT_ASSET = "GOLD"
DEFAULT_COT_PCT = 0.90       # a bar fires when its (lagged) net-positioning is at/above this quantile of its past
DEFAULT_HOLD_BARS = 10       # ~2 weeks (COT is weekly), then flat
DEFAULT_RELEASE_LAG = 4      # report Tuesday + 4 calendar days -> usable the Monday after the Friday release
DEFAULT_MIN_HISTORY = 30
DEFAULT_FEE = 0.0005

BAR_MINUTES = 1440
COT_DIR = "cot"
_UTC = datetime.timezone.utc


# --- COT join: release-lagged ------------------------------------------------------------------------


def _load_cot(asset):
    path = f"{COT_DIR}/{asset}.json"
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def cot_values(asset, timestamps, release_lag_days=DEFAULT_RELEASE_LAG):
    """Per-bar net-positioning value (net / open interest) — the latest COT report whose (report date +
    release_lag_days) is at/before each bar, ``None`` before the first. The lag is the publication guard: the
    Tuesday report is public only from the following Friday, so a report dated D is invisible until D+lag.
    ``timestamps`` are epoch ms (assumed ascending, as bars are)."""
    table = _load_cot(asset)
    lag = int(release_lag_days)
    entries = []
    for date, rec in table.items():
        oi, net = rec.get("oi"), rec.get("net")
        if oi is None or net is None or oi <= 0:
            continue
        eff = datetime.date.fromisoformat(date) + datetime.timedelta(days=lag)
        eff_ms = int(datetime.datetime(eff.year, eff.month, eff.day, tzinfo=_UTC).timestamp() * 1000)
        entries.append((eff_ms, net / oi))
    entries.sort()
    out = []
    i, cur = 0, None
    for ts in timestamps:
        while i < len(entries) and entries[i][0] <= ts:
            cur = entries[i][1]
            i += 1
        out.append(cur)
    return out


# --- extreme signal: extreme of the PAST-ONLY positioning distribution -------------------------------


def cot_index_series(values, window, min_history=DEFAULT_MIN_HISTORY):
    """The TRAILING-window COT INDEX (Williams min-max) value in [0,100] per bar: 100*(value[t]-min)/(max-min)
    over the trailing ``window`` bars ENDING at t (strictly past-or-present, so past-only). ``None`` until
    ``min_history`` values exist or where the window is degenerate. This is the raw regime-adaptive positioning
    score; ``_cot_index_sides`` thresholds it, and the cross-sectional probe ranks commodities on it."""
    n = len(values)
    out = [None] * n
    for t in range(n):
        if t < int(min_history) or values[t] is None:
            continue
        seg = [v for v in values[max(0, t - int(window) + 1): t + 1] if v is not None]
        if len(seg) < int(min_history):
            continue
        mn, mx = min(seg), max(seg)
        if mx <= mn:
            continue
        out[t] = 100.0 * (values[t] - mn) / (mx - mn)
    return out


def _cot_index_sides(values, invert, pct, window, min_history):
    """Per-bar side from the trailing COT INDEX: long at the top (index >= pct*100) / short at the bottom
    (index <= (1-pct)*100); ``invert`` flips (contrarian). Unlike the expanding quantile, the trailing
    normalisation adapts to regime and cannot be anchored unreachable by an old positioning mania."""
    idx = cot_index_series(values, window, min_history)
    hi, lo = float(pct) * 100.0, (1.0 - float(pct)) * 100.0
    out = [None] * len(idx)
    for t in range(len(idx)):
        if idx[t] is None:
            continue
        side = 1 if idx[t] >= hi else (-1 if idx[t] <= lo else None)
        if side is not None:
            out[t] = float(-side if invert else side)
    return out


def _flow_values(values, lag):
    """The CHANGE (weekly flow) in positioning: out[t] = value[t] - value[t-lag] (``None`` where either end is
    absent). This turns the LEVEL series into a FLOW series — which way managed money is MOVING, a distinct
    signal from how extreme its level is — before the same extreme/side machinery is applied."""
    n = len(values)
    lag = int(lag)
    out = [None] * n
    for t in range(n):
        if t >= lag and values[t] is not None and values[t - lag] is not None:
            out[t] = values[t] - values[t - lag]
    return out


def _signal_values(values, cfg):
    """The series the extreme/side is computed on: the raw positioning LEVEL, or its weekly CHANGE (flow) when
    ``cot_flow_lag`` > 0."""
    lag = int(cfg.get("cot_flow_lag", 0))
    return _flow_values(values, lag) if lag > 0 else values


def cot_sides(timestamps, values, signal, cot_pct, min_history=DEFAULT_MIN_HISTORY, index_window=0):
    """Per-bar entry side (+1 / -1 / ``None``) from the positioning extreme. A top-tail crowded-long extreme is a
    SHORT for cot_contrarian (the crowd unwinds) / a LONG for cot_momentum (ride it), and a bottom-tail extreme
    the reverse. Default: extreme of the EXPANDING past-only distribution (``past_only_extreme_sides``); when
    ``index_window`` > 0, the TRAILING-window COT Index instead (regime-adaptive). This only maps the two signal
    names to the invert flag."""
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    invert = signal == "cot_contrarian"
    if int(index_window) > 0:
        return _cot_index_sides(values, invert=invert, pct=float(cot_pct), window=int(index_window), min_history=int(min_history))
    return past_only_extreme_sides(values, invert=invert, pct=float(cot_pct), min_history=int(min_history))


def positions_from_bars(timestamps, closes, values, cfg, start_index=0):
    """Decision-bar position vector for the COT signal. Positions depend ONLY on the (release-lagged) positioning
    values + timestamps, never on ``closes``; the per-bar side is fed through intraday's hold-then-flat builder,
    so an extreme at bar t occupies t+1..t+hold_bars, new extremes inside the hold window are ignored, and
    extremes before ``start_index`` are suppressed (the formation lookback warms the positioning distribution)."""
    n = len(timestamps)
    if n == 0:
        return np.zeros(0)
    sides = cot_sides(
        timestamps, _signal_values(values, cfg), str(cfg.get("signal", DEFAULT_SIGNAL)),
        float(cfg.get("cot_pct", DEFAULT_COT_PCT)), int(cfg.get("min_history", DEFAULT_MIN_HISTORY)),
        index_window=int(cfg.get("cot_index_window", 0)),
    )
    return _build_positions(n, lambda t: sides[t], int(cfg.get("hold_bars", DEFAULT_HOLD_BARS)), start_index)


# --- the run contract ---------------------------------------------------------------------------------


def run(cfg):
    """Run one COT cell and return a trainer-contract RunSummary. The month before the test window is a formation
    lookback for the past-only positioning distribution; only the TEST span is accounted. The objective is the
    per-step OOS Sharpe; return_vs_hold_pct is versus a fee-charged buy-and-hold. n_signals (COT entries inside
    the accounted span) is emitted so a thin count is visible; signal_expectancy is the exposure-neutral decider
    for this sparse book."""
    asset = str(cfg.get("asset", DEFAULT_ASSET))
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    fee = float(cfg.get("transaction_fee", DEFAULT_FEE))
    release_lag = int(cfg.get("release_lag_days", DEFAULT_RELEASE_LAG))

    train_pairs, test_pairs, meta = resolve_walk_forward_window(cfg)
    load_pairs = [_prior_month(test_pairs[0])] + list(test_pairs)
    # warm the positioning distribution with the training span too, so the past-only quantile is populated
    load_pairs = list(train_pairs) + list(test_pairs) if train_pairs else load_pairs
    bars = _load_bars(asset, load_pairs, BAR_MINUTES)
    closes = bars["close"]
    timestamps = bars["timestamp"]
    values = cot_values(asset, timestamps, release_lag)

    test_from_ms = _month_start_ms(meta["test_from"])
    start_index = next((i for i, ts in enumerate(timestamps) if ts >= test_from_ms), len(timestamps))

    pos = positions_from_bars(timestamps, closes, values, cfg, start_index=start_index)
    closes_t = closes[start_index:]
    pos_t = pos[start_index:]
    ts_t = timestamps[start_index:]

    equity = backtest(closes_t, pos_t, fee)
    n_rt = round_trips(pos_t)
    total_return_pct = (equity[-1] - 1.0) * 100.0 if len(equity) >= 1 else 0.0
    turnover = float(np.abs(np.diff(np.concatenate([[0.0], np.asarray(pos_t, dtype=float)]))).sum())
    sides_t = cot_sides(timestamps, _signal_values(values, cfg), signal, float(cfg.get("cot_pct", DEFAULT_COT_PCT)), int(cfg.get("min_history", DEFAULT_MIN_HISTORY)), index_window=int(cfg.get("cot_index_window", 0)))[start_index:]
    n_signals = sum(1 for s in sides_t if s is not None)

    metrics = {
        "total_return_pct": _finite(total_return_pct),
        "baseline": 0.0,
        "n_trades": n_rt,
        "n_signals": n_signals,
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
        hstats = _oos_stats(_hold_equity(prices, round_trip))
        if "oos_sharpe" in hstats:
            benchmark["hold_sharpe"] = hstats["oos_sharpe"]
        metrics["hold_return_pct"] = benchmark["hold_return_pct"]
        metrics["return_vs_hold_pct"] = _finite(total_return_pct - benchmark["hold_return_pct"])
        metrics["hold_net_of_fees"] = True
    metrics.setdefault("return_vs_hold_pct", 0.0)

    objective = metrics.get("oos_sharpe", 0.0)
    stored_cfg = dict(cfg)
    summary = {
        "objective": _finite(objective),
        "metrics": metrics,
        "health": {"status": "ok", "flags": []},
        "config": stored_cfg,
        "dataset": {
            "asset": asset,
            "timeframe": f"{BAR_MINUTES}m",
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
    parser = argparse.ArgumentParser(description="CFTC COT managed-money positioning-extreme backtest")
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
        f"n_signals={m.get('n_signals', 0)} signal_expectancy={m.get('signal_expectancy', 0.0):.4f} "
        f"return_vs_hold_pct={m.get('return_vs_hold_pct', 0.0):.4f} -> {args.summary_out}"
    )


if __name__ == "__main__":
    main()
