"""Attention probe — the first SENTIMENT/attention stream (B3), the cheap calibration.

Wikipedia daily pageviews for an asset's article (Bitcoin / Ethereum / Solana) proxy retail ATTENTION — a free,
no-key, per-asset, daily series mined to wiki-pageviews/<asset>.json. This probe asks the honest B3 question:
does an attention SURGE (pageviews in the top tail of their own past) predict the next day's move — ride it
(attention_momentum: the FOMO continues) or fade it (attention_reversion: the hype day is a local top)? The
well-known trap is that attention tracks VOLATILITY, not direction; this measures whether any tradeable
direction survives cost, DSR-corrected.

It is deliberately thin: the extreme-of-past rule is the shared signal_extremes core the funding/order-flow
probes use, and the backtest MECHANICS are intraday.py's, so every gate, lens and scorecard reads an attention
cell unchanged. The one surface new here is the pageview JOIN, which carries the leakage trap pinned in
test_attention.py: Wikimedia publishes a day's count only AFTER that day, so a crypto bar dated D reads
pageviews[D - lag_days] (lag_days>=1) — never day D's own still-unpublished count — making the traded return two
days after the pageview day. "Is attention extreme?" is past-only (delegated), the position applies to the next
bar, and positions never read price. Deterministic and model-free (the seed is contract-only).
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

# ride the attention surge (attention_momentum) vs fade it (attention_reversion). An unknown value is refused
# so a typo can never file one arm's result under the other's label.
SIGNALS = ("attention_momentum", "attention_reversion")
DEFAULT_SIGNAL = "attention_momentum"

DEFAULT_ASSET = "BTCUSDT"
DEFAULT_ATTENTION_PCT = 0.90  # a bar fires when its (lagged) pageviews are at/above this quantile of the past
DEFAULT_HOLD_BARS = 2
DEFAULT_LAG_DAYS = 1          # publication lag: a bar dated D reads pageviews[D - lag_days] (D's count is unpublished)
DEFAULT_MIN_HISTORY = 30
DEFAULT_FEE = 0.001

BAR_MINUTES = 1440           # daily decisions — the cadence pageviews arrive at
PAGEVIEWS_DIR = "wiki-pageviews"


# --- pageview join: publication-lagged -------------------------------------------------------------


def _load_pageviews(asset):
    path = f"{PAGEVIEWS_DIR}/{asset}.json"
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def attention_values(asset, timestamps, lag_days=DEFAULT_LAG_DAYS):
    """Per-bar attention value: the pageviews of the calendar day ``lag_days`` before each bar's date (``None``
    when absent). The lag is the publication guard — a day's count is published only after that day, so a bar
    dated D uses day D-1's count at the earliest, never its own. ``timestamps`` are epoch ms (daily bars align
    to UTC midnight)."""
    pv = _load_pageviews(asset)
    lag = int(lag_days)
    if lag < 1:
        raise SystemExit(f"lag_days must be >=1: a day's pageviews are published only AFTER that day, so a lag<1 join would read the bar's own (or a future) still-unpublished count — a lookahead. got {lag_days!r}")
    out = []
    for ts in timestamps:
        d = datetime.datetime.utcfromtimestamp(int(ts) / 1000.0).date() - datetime.timedelta(days=lag)
        v = pv.get(d.isoformat())
        out.append(float(v) if v is not None else None)
    return out


# --- surge signal: extreme of the PAST-ONLY pageview distribution ------------------------------------


def attention_sides(timestamps, values, signal, attention_pct, min_history=DEFAULT_MIN_HISTORY):
    """Per-bar entry side (+1 / -1 / ``None``) from the attention extreme at each bar. The lagged pageview is
    the scalar; a top-tail surge versus its own past is a long for attention_momentum / short for
    attention_reversion (and a bottom-tail lull the reverse). The past-only quantile discipline and
    None-handling live in ``past_only_extreme_sides``; this only maps the two signal names to the invert flag."""
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    return past_only_extreme_sides(
        values, invert=(signal == "attention_reversion"), pct=float(attention_pct), min_history=int(min_history)
    )


def positions_from_bars(timestamps, closes, values, cfg, start_index=0):
    """Decision-bar position vector for the attention signal. Positions depend ONLY on the (lagged) attention
    values + timestamps, never on ``closes``; the per-bar side is fed through intraday's hold-then-flat builder,
    so a surge at bar t occupies t+1..t+hold_bars, new surges inside the hold window are ignored, and surges
    before ``start_index`` are suppressed (the formation lookback warms the attention distribution)."""
    n = len(timestamps)
    if n == 0:
        return np.zeros(0)
    sides = attention_sides(
        timestamps,
        values,
        str(cfg.get("signal", DEFAULT_SIGNAL)),
        float(cfg.get("attention_pct", DEFAULT_ATTENTION_PCT)),
        int(cfg.get("min_history", DEFAULT_MIN_HISTORY)),
    )
    return _build_positions(n, lambda t: sides[t], int(cfg.get("hold_bars", DEFAULT_HOLD_BARS)), start_index)


# --- the run contract ---------------------------------------------------------------------------------


def run(cfg):
    """Run one attention cell and return a trainer-contract RunSummary. The month before the test window is a
    formation lookback for the past-only pageview distribution; only the TEST span is accounted. The objective
    is the per-step OOS Sharpe; return_vs_hold_pct is versus a fee-charged buy-and-hold of the same asset.
    n_surges (attention entries inside the accounted span) is emitted so a thin count is visible."""
    asset = str(cfg.get("asset", DEFAULT_ASSET))
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    fee = float(cfg.get("transaction_fee", DEFAULT_FEE))
    lag_days = int(cfg.get("lag_days", DEFAULT_LAG_DAYS))

    _, test_pairs, meta = resolve_walk_forward_window(cfg)
    load_pairs = [_prior_month(test_pairs[0])] + list(test_pairs)
    bars = _load_bars(asset, load_pairs, BAR_MINUTES)
    closes = bars["close"]
    timestamps = bars["timestamp"]
    values = attention_values(asset, timestamps, lag_days)

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
    sides_t = attention_sides(timestamps, values, signal, float(cfg.get("attention_pct", DEFAULT_ATTENTION_PCT)), int(cfg.get("min_history", DEFAULT_MIN_HISTORY)))[start_index:]
    n_surges = sum(1 for s in sides_t if s is not None)

    metrics = {
        "total_return_pct": _finite(total_return_pct),
        "baseline": 0.0,
        "n_trades": n_rt,
        "n_surges": n_surges,
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
    parser = argparse.ArgumentParser(description="Wikipedia-attention surge backtest")
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
        f"n_surges={m.get('n_surges', 0)} trades_per_day={m.get('trades_per_day', 0.0):.4f} "
        f"signal_expectancy={m.get('signal_expectancy', 0.0):.4f} "
        f"return_vs_hold_pct={m.get('return_vs_hold_pct', 0.0):.4f} -> {args.summary_out}"
    )


if __name__ == "__main__":
    main()
