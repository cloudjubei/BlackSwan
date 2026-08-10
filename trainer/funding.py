"""Perpetual-funding probe — the first NON-PRICE stream in the program.

Every trading line before this one read only PRICE (and derivatives of it) and found no directional edge
that clears cost. Perp funding is different in kind: it is the periodic payment between longs and shorts
that tethers the perpetual to spot, and its sign/size is a positioning/sentiment gauge that the spot price
series does not contain. Extreme-positive funding means longs are crowded (paying shorts); extreme-negative
means shorts are crowded. This probe asks the honest B1 question: does conditioning an 8h-cadence directional
trade on a funding EXTREME — fading it (contrarian) or riding it (momentum) — beat buy-and-hold NET of the
~0.2% crypto round trip, walk-forward and multiple-testing-corrected?

It is deliberately thin: the backtest MECHANICS are intraday.py's, reused verbatim (resample 1m -> decision
bars, the hold-then-flat position builder, both-way fees, per-trade edge, trades_per_day) so every existing
gate, lens and scorecard reads a funding cell unchanged. Funding settles on the 8h grid (00:00/08:00/16:00
UTC), which is exactly where a 480-minute resample lands, so a funding value joins its decision bar by
TIMESTAMP. The two surfaces this module adds beyond intraday are the funding JOIN and the funding SIGNAL, and
both carry lookahead traps pinned in test_funding.py before a number is emitted:

* The join is by timestamp with a tolerance, NOT by index or a floored bucket. Rows carry sub-second jitter
  (kept) and, under stress, binance inserts off-cycle 4h fundings that sit mid-8h-bar (dropped — a decision
  at an 8h close never observed them). load_funding enforces this.

* "Is this funding extreme?" is judged against the funding distribution known strictly BEFORE the bar — the
  same past-only-quantile discipline the intraday regime gate uses. A whole-sample quantile would let a
  level that is extreme-versus-the-sample but ordinary-versus-its-own-past fabricate a signal.

* The position implied by the funding that settled at bar t is applied to bars t+1.. (never bar t's own
  interval), so the position pipeline is time-prefix causal in funding. Positions never read price at all —
  only the funding signal does — a stronger guarantee than the breakout arm needs.

Deterministic and model-free (the seed is contract-only). The decision cadence IS the funding cadence, so
bar width is fixed at 480m rather than swept.
"""

import argparse
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

# The two competing readings of a funding extreme. contrarian FADES it (crowded positioning mean-reverts);
# momentum RIDES it (funding reflects genuine, persistent demand). An unknown value is refused so a typo
# can never file one arm's result under the other's label.
SIGNALS = ("funding_contrarian", "funding_momentum")
DEFAULT_SIGNAL = "funding_contrarian"

DEFAULT_ASSET = "BTCUSDT"
DEFAULT_FUNDING_PCT = 0.80   # a bar fires when funding is at/above this quantile (or at/below 1-this) of its past
DEFAULT_HOLD_BARS = 1        # 8h intervals a position is held after entry, then flat
DEFAULT_MIN_HISTORY = 30     # funding observations required before any bar can be judged extreme
DEFAULT_FEE = 0.001          # per side; the round trip (1-fee)^2 is ~0.2%, the crypto cost floor

FUNDING_BAR_MINUTES = 480          # 8h — the funding settlement cadence; not a lever
FUNDING_BUCKET_MS = 28_800_000     # 8h in ms
FUNDING_JOIN_TOL_MS = 300_000      # a funding row within 5 min of an 8h boundary settles AT it; further off is off-cycle
FUNDING_DIR = "binance-funding"


# --- funding data: load + tolerance join to the 8h grid ----------------------------------------------


def load_funding(path):
    """Load a mined funding file into a ``{bucket_ms: rate}`` map keyed to the 8h settlement grid.

    Each row's ``fundingTime`` is snapped to its nearest 8h bucket, but ONLY when it lands within
    ``FUNDING_JOIN_TOL_MS`` of that boundary — the on-grid settlements (sub-second jitter) are kept; the
    off-cycle 4h fundings binance inserts under stress sit ~4h from any boundary and are DROPPED, because a
    decision taken at an 8h bar close never observed them. Rates arrive as strings and are cast here. A
    missing file yields an empty map rather than raising, so an un-mined asset simply trades flat."""
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        rows = json.load(fh)
    out = {}
    for r in rows:
        ft = int(r["fundingTime"])
        bucket = round(ft / FUNDING_BUCKET_MS) * FUNDING_BUCKET_MS
        if abs(ft - bucket) <= FUNDING_JOIN_TOL_MS:
            try:
                out[bucket] = float(r["fundingRate"])
            except (TypeError, ValueError):
                continue
    return out


# --- funding signal: extreme versus the PAST-ONLY funding distribution -------------------------------


def funding_sides(timestamps, funding_map, signal, funding_pct, min_history=DEFAULT_MIN_HISTORY):
    """Per-bar entry side (+1 / -1 / ``None``) from the funding extreme at each bar, judged past-only.

    The funding that settled at each bar's timestamp is the per-bar scalar; a crowded-LONG extreme (high
    funding) versus its own past is a short for contrarian / long for momentum, and vice versa. The past-only
    quantile discipline and None-handling live in ``past_only_extreme_sides``; this only joins funding to bars
    and maps the two signal names to the invert flag (contrarian fades the crowd, momentum rides it)."""
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    values = [funding_map.get(ts) for ts in timestamps]
    return past_only_extreme_sides(
        values, invert=(signal == "funding_contrarian"), pct=float(funding_pct), min_history=int(min_history)
    )


def positions_from_bars(timestamps, closes, funding_map, cfg, start_index=0):
    """Decision-bar position vector for the funding signal. Positions depend ONLY on funding + timestamps
    (never on ``closes`` — passed solely so the signature mirrors the intraday builder and a caller can hold
    the aligned price for the backtest). The per-bar side is fed through intraday's hold-then-flat builder,
    so a fire at bar t occupies t+1..t+hold_bars, new fires inside the hold window are ignored, and entries
    before ``start_index`` are suppressed (the formation lookback warms the funding distribution)."""
    n = len(timestamps)
    if n == 0:
        return np.zeros(0)
    sides = funding_sides(
        timestamps,
        funding_map,
        str(cfg.get("signal", DEFAULT_SIGNAL)),
        float(cfg.get("funding_pct", DEFAULT_FUNDING_PCT)),
        int(cfg.get("min_history", DEFAULT_MIN_HISTORY)),
    )
    return _build_positions(n, lambda t: sides[t], int(cfg.get("hold_bars", DEFAULT_HOLD_BARS)), start_index)


# --- the run contract ---------------------------------------------------------------------------------


def _funding_path(asset):
    return f"{FUNDING_DIR}/{asset}-funding.json"


def run(cfg):
    """Run one funding cell and return a trainer-contract RunSummary.

    The month before the test window is loaded as a formation lookback to warm the past-only funding
    distribution; only the TEST span is accounted, exactly as the daily and intraday lines report. The
    objective is the per-step OOS Sharpe; return_vs_hold_pct is versus a fee-charged buy-and-hold of the
    same asset over the same span. funding_coverage (fraction of accounted bars carrying a joined funding
    value) is emitted so a broken join surfaces as a metric instead of a silent all-flat book."""
    asset = str(cfg.get("asset", DEFAULT_ASSET))
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    fee = float(cfg.get("transaction_fee", DEFAULT_FEE))

    _, test_pairs, meta = resolve_walk_forward_window(cfg)
    load_pairs = [_prior_month(test_pairs[0])] + list(test_pairs)
    bars = _load_bars(asset, load_pairs, FUNDING_BAR_MINUTES)
    closes = bars["close"]
    timestamps = bars["timestamp"]
    funding_map = load_funding(_funding_path(asset))

    test_from_ms = _month_start_ms(meta["test_from"])
    start_index = next((i for i, ts in enumerate(timestamps) if ts >= test_from_ms), len(timestamps))

    pos = positions_from_bars(timestamps, closes, funding_map, cfg, start_index=start_index)
    closes_t = closes[start_index:]
    pos_t = pos[start_index:]
    ts_t = timestamps[start_index:]

    equity = backtest(closes_t, pos_t, fee)
    n_rt = round_trips(pos_t)
    total_return_pct = (equity[-1] - 1.0) * 100.0 if len(equity) >= 1 else 0.0
    turnover = float(np.abs(np.diff(np.concatenate([[0.0], np.asarray(pos_t, dtype=float)]))).sum())
    covered = sum(1 for ts in ts_t if _is_num(funding_map.get(ts)))

    metrics = {
        "total_return_pct": _finite(total_return_pct),
        "baseline": 0.0,
        "n_trades": n_rt,
        "realized_cost_bps": _finite(turnover * fee * 10000),
        "final_net_worth": _finite(equity[-1]) if equity else 1.0,
        "trades_per_day": trades_per_day(n_rt, ts_t[0], ts_t[-1]) if len(ts_t) >= 2 else 0.0,
        "funding_coverage": _finite(covered / len(ts_t)) if ts_t else 0.0,
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
            "timeframe": f"{FUNDING_BAR_MINUTES}m",
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
    parser = argparse.ArgumentParser(description="Perpetual-funding conditional-edge backtest")
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
        f"funding_coverage={m.get('funding_coverage', 0.0):.3f} "
        f"return_vs_hold_pct={m.get('return_vs_hold_pct', 0.0):.4f} -> {args.summary_out}"
    )


if __name__ == "__main__":
    main()
