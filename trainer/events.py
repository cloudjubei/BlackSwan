"""Macro-event probe — event-CONDITIONED trading (the B2 line, exogenous triggers).

The intraday probe measured UNCONDITIONAL short-horizon direction as null. This line asks a narrower,
genuinely exogenous question: in the window right after a SCHEDULED macro release (CPI, jobs, retail, PCE,
GDP), does the crypto reaction PERSIST (event_drift) or REVERSE (event_fade) far enough to clear the ~0.2%
round trip? A scheduled release is external information the price could not have fully known a bar earlier, so
if any short-horizon drift is real anywhere, the minutes after a release are where it should live. The probe
trades ONLY in those windows (few trades/day), the "recognise the condition, trade only then" discipline.

It is deliberately thin: the release TIMING is the shared point-in-time machinery (pit_fusion — release date +
the series' publish wall-clock, DST-aware, the same leakage guard the context fusion uses), and the backtest
MECHANICS are intraday.py's, reused verbatim. The surfaces new here, pinned in test_events.py before a number
is emitted:

* an event fires at the series' real RELEASE datetime, NEVER the reference period it describes (stamping at
  refPeriod is the classic macro look-ahead). Only genuine DISCRETE releases are events — the daily rate
  series (DFEDTARU/DFF/DGS10/…) are excluded, since a value every calendar day is not a market event. Events
  outside the accounted span are dropped, and coincident releases across series collapse to one event.

* the reaction bar is the FIRST decision bar at/after the release; its return is known at that bar's close and
  the position it implies is applied to the NEXT bars, never the reaction bar's own return — time-prefix
  causal. Event times are fixed by the macro calendar and never depend on price.

Deterministic and model-free (the seed is contract-only).
"""

import argparse
import bisect
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
    _sign,
    _trade_edges,
    backtest,
    bar_returns,
    round_trips,
    trades_per_day,
)
from trainer.pit_fusion import publish_time_for, release_datetime_ms
from trainer.summary import (
    _capture_stats,
    _finite,
    _hold_equity,
    _max_drawdown_pct,
    _oos_stats,
    _provenance_fingerprint,
)
from trainer.walk_forward import resolve_walk_forward_window

# ride the post-release reaction (information drift) vs fade it (overreaction correction). An unknown value is
# refused so a typo can never file one arm's result under the other's label.
SIGNALS = ("event_drift", "event_fade")
DEFAULT_SIGNAL = "event_drift"

# Event groups over GENUINE discrete releases only — the daily rate series (DFEDTARU/DFF/DGS10/T10Y2Y/DFII10)
# are intentionally absent: a value every calendar day is not a market event and would fire every day. FOMC
# needs a real meeting calendar (not a daily rate print), so it is deferred until that is mined.
EVENT_GROUPS = {
    "cpi": ("CPIAUCNS", "CPIAUCSL"),
    "jobs": ("PAYEMS", "ICSA", "UNRATE"),
    "macro_all": ("CPIAUCNS", "CPIAUCSL", "PAYEMS", "ICSA", "UNRATE", "RSAFS", "PCEPILFE", "GDPC1"),
}

DEFAULT_ASSET = "BTCUSDT"
DEFAULT_BAR = 60
DEFAULT_EVENT_GROUP = "macro_all"
DEFAULT_HOLD_BARS = 4
DEFAULT_FEE = 0.001
MACRO_DIR = "macro"


# --- event timing: release-based, DST-aware, deduped ------------------------------------------------


def event_ms_from_observations(observations, series_id, lo_ms, hi_ms):
    """Release timestamps (epoch ms) for ``observations`` of ``series_id`` that fall in ``[lo_ms, hi_ms]``.

    Each observation is stamped at its RELEASE datetime (``releaseDate`` at the series' publish wall-clock,
    DST-aware, via pit_fusion) — never its reference period. Observations without a ``releaseDate`` are
    skipped; the window filter keeps only events inside the loaded span."""
    pt = publish_time_for(series_id)
    out = []
    for obs in observations:
        rd = obs.get("releaseDate")
        if rd is None:
            continue
        ms = release_datetime_ms(rd, pt)
        if lo_ms <= ms <= hi_ms:
            out.append(ms)
    return out


def load_event_ms(group, lo_ms, hi_ms):
    """Sorted, de-duplicated release timestamps for an event ``group`` over ``[lo_ms, hi_ms]``. Coincident
    releases across the group's series (e.g. CPI NSA + SA drop together) collapse to one event; a series file
    absent on disk is skipped. An unknown group is refused rather than silently trading nothing."""
    series = EVENT_GROUPS.get(group)
    if series is None:
        raise SystemExit(f"unknown event_group {group!r}; choose one of {sorted(EVENT_GROUPS)}")
    instants = set()
    for s in series:
        path = f"{MACRO_DIR}/{s}.json"
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            obs = json.load(fh)
        instants.update(event_ms_from_observations(obs, s, lo_ms, hi_ms))
    return sorted(instants)


# --- reaction gate: first bar at/after the release, ride vs fade -------------------------------------


def reaction_sides(timestamps, closes, event_ms, signal):
    """Per-bar entry side (+1 / -1 / ``None``) from the post-release reaction, one per event.

    For each event instant, the reaction bar is the FIRST decision bar with ``timestamp >= event`` (the first
    bar to close after the release). Its return — known at that bar's close — is ridden (event_drift) or faded
    (event_fade); a zero or undefined reaction leaves the bar flat. ``timestamps`` is assumed sorted (bars are
    time-ordered). Only the reaction bars carry a side; every other bar is ``None``."""
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    fade = signal == "event_fade"
    n = len(timestamps)
    sides = [None] * n
    returns = bar_returns(closes)
    for e in event_ms:
        b = bisect.bisect_left(timestamps, e)  # first index with timestamps[b] >= e
        if b < n and _is_num(returns[b]):
            s = _sign(returns[b])
            if s != 0.0:
                sides[b] = (-s if fade else s)
    return sides


def positions_from_events(timestamps, closes, event_ms, cfg, start_index=0):
    """Decision-bar position vector for the event signal. The per-event reaction side is fed through intraday's
    hold-then-flat builder, so a reaction at bar b occupies b+1..b+hold_bars (never bar b's own return), a new
    event inside the hold window is ignored, and reactions before ``start_index`` are suppressed (the formation
    lookback carries events that opened before the accounted span)."""
    n = len(timestamps)
    if n == 0:
        return np.zeros(0)
    sides = reaction_sides(timestamps, closes, event_ms, str(cfg.get("signal", DEFAULT_SIGNAL)))
    return _build_positions(n, lambda t: sides[t], int(cfg.get("hold_bars", DEFAULT_HOLD_BARS)), start_index)


# --- the run contract ---------------------------------------------------------------------------------


def run(cfg):
    """Run one macro-event cell and return a trainer-contract RunSummary.

    The month before the test window is loaded as a formation lookback (so an event opening just before the
    accounted span is handled), but only the TEST span is accounted, exactly as the daily and intraday lines
    report. The objective is the per-step OOS Sharpe; return_vs_hold_pct is versus a fee-charged buy-and-hold
    of the same asset over the same span. n_events (releases inside the accounted span) is emitted so a thin
    event count is visible next to the result."""
    asset = str(cfg.get("asset", DEFAULT_ASSET))
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    bar_minutes = int(cfg.get("bar", DEFAULT_BAR))
    group = str(cfg.get("event_group", DEFAULT_EVENT_GROUP))
    fee = float(cfg.get("transaction_fee", DEFAULT_FEE))

    _, test_pairs, meta = resolve_walk_forward_window(cfg)
    load_pairs = [_prior_month(test_pairs[0])] + list(test_pairs)
    bars = _load_bars(asset, load_pairs, bar_minutes)
    closes = bars["close"]
    timestamps = bars["timestamp"]

    event_ms = load_event_ms(group, timestamps[0], timestamps[-1]) if timestamps else []

    test_from_ms = _month_start_ms(meta["test_from"])
    start_index = next((i for i, ts in enumerate(timestamps) if ts >= test_from_ms), len(timestamps))

    pos = positions_from_events(timestamps, closes, event_ms, cfg, start_index=start_index)
    closes_t = closes[start_index:]
    pos_t = pos[start_index:]
    ts_t = timestamps[start_index:]

    equity = backtest(closes_t, pos_t, fee)
    n_rt = round_trips(pos_t)
    total_return_pct = (equity[-1] - 1.0) * 100.0 if len(equity) >= 1 else 0.0
    turnover = float(np.abs(np.diff(np.concatenate([[0.0], np.asarray(pos_t, dtype=float)]))).sum())
    n_events = sum(1 for e in event_ms if e >= test_from_ms)

    metrics = {
        "total_return_pct": _finite(total_return_pct),
        "baseline": 0.0,
        "n_trades": n_rt,
        "n_events": n_events,
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
    parser = argparse.ArgumentParser(description="Macro-event conditional-edge backtest")
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
        f"n_events={m.get('n_events', 0)} trades_per_day={m.get('trades_per_day', 0.0):.4f} "
        f"return_vs_hold_pct={m.get('return_vs_hold_pct', 0.0):.4f} -> {args.summary_out}"
    )


if __name__ == "__main__":
    main()
