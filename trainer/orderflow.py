"""Order-flow (taker-imbalance) probe — the second NON-PRICE stream, and the cheapest possible one.

The binance 1m klines BlackSwan already holds carry the aggressor split for free: ``asset_volume_taker_base``
is the base volume where the TAKER was the buyer (an aggressive market buy), so aggressive selling is
``volume - taker_buy`` and the per-bar signed imbalance is (2*taker_buy - volume)/volume in [-1, +1] — net
aggressive buying above 0, net selling below. No mining is needed; the stream is on disk. This probe asks
whether conditioning a directional trade on an EXTREME of that imbalance beats buy-and-hold net of the ~0.2%
round trip: flow_momentum RIDES the extreme (informed aggressor flow persists), flow_contrarian FADES it
(aggressors overpay the spread and get run over).

It is deliberately thin. The extreme-of-past rule is the shared ``signal_extremes.past_only_extreme_sides`` the
funding probe uses, and the backtest MECHANICS are intraday.py's, reused verbatim (hold-then-flat positions,
both-way fees, per-trade edge, trades_per_day) so every existing gate, lens and scorecard reads a flow cell
unchanged. The surfaces new here are the flow aggregation and the imbalance sign, each pinned in
test_orderflow.py before a number is emitted:

* imbalance is built from ONLY a bar's own minutes (no straddle) and is the SIGNED aggressor ratio — an
  all-buy bar is +1, an all-sell bar is -1, a balanced bar is 0, a zero-volume bar is undefined (None, never a
  fake 0). A sign slip or a total-vs-taker mix-up would invert or flatten the whole signal silently.

* the imbalance measured over bar t is known at t's close and the position it implies is applied to bars
  t+1.. — never bar t's own return — so the position pipeline is time-prefix causal in flow, and positions
  never read price at all (only the imbalance does).

Deterministic and model-free (the seed is contract-only). Unlike funding (fixed 8h), taker imbalance
aggregates to any bar width, so ``bar`` is a lever — smaller bars trade more often and pay the cost floor more
often, the central tension. flow_coverage (fraction of accounted bars with a defined imbalance) is emitted so a
broken aggregation surfaces as a metric, not a silent all-flat book.
"""

import argparse
import json
import os

import numpy as np

from trainer import data_catalog
from trainer.intraday import (
    _build_positions,
    _is_num,
    _iso_from_ms,
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

# ride the aggressor flow (informed demand persists) vs fade it (aggressors overpay). An unknown value is
# refused so a typo can never file one arm's result under the other's label.
SIGNALS = ("flow_momentum", "flow_contrarian")
DEFAULT_SIGNAL = "flow_momentum"

DEFAULT_ASSET = "BTCUSDT"
DEFAULT_BAR = 60           # decision-bar width in minutes (1m aggregated up to this)
DEFAULT_FLOW_PCT = 0.80    # a bar fires when its imbalance is at/above this quantile (or at/below 1-this) of its past
DEFAULT_HOLD_BARS = 1
DEFAULT_MIN_HISTORY = 30
DEFAULT_FEE = 0.001


# --- flow aggregation: 1m taker split -> decision-bar signed imbalance --------------------------------


def flow_bars(rows, bar_minutes):
    """Aggregate contiguous 1-minute rows into decision bars carrying signed taker imbalance.

    Each bar is bucketed by ``timestamp // (bar_minutes * 60_000)`` (epoch-aligned, so a bar never straddles a
    month boundary and per-file aggregation then concatenation equals aggregating the whole span). A bar's
    close is its last minute's close; its ``volume`` and ``taker_buy`` are the sums over ONLY its own minutes;
    its ``imbalance`` is (2*taker_buy - volume)/volume in [-1, +1], or ``None`` when the bar has no volume (an
    undefined imbalance must never masquerade as a balanced 0). Volumes arrive as strings and are cast here."""
    out = {"timestamp": [], "close": [], "volume": [], "taker_buy": [], "imbalance": []}
    if not rows:
        return out
    width = int(bar_minutes) * 60_000

    def flush(cur):
        out["timestamp"].append(cur["ts"])
        out["close"].append(cur["c"])
        out["volume"].append(cur["v"])
        out["taker_buy"].append(cur["tb"])
        out["imbalance"].append((2.0 * cur["tb"] - cur["v"]) / cur["v"] if cur["v"] > 0 else None)

    cur = None
    for row in rows:
        ts = int(row["timestamp"])
        bucket = ts // width
        c = float(row["price"])
        v = float(row.get("volume", 0.0) or 0.0)
        tb = float(row.get("asset_volume_taker_base", 0.0) or 0.0)
        if cur is None or bucket != cur["bucket"]:
            if cur is not None:
                flush(cur)
            cur = {"bucket": bucket, "ts": bucket * width, "c": c, "v": v, "tb": tb}
        else:
            cur["c"] = c
            cur["v"] += v
            cur["tb"] += tb
    if cur is not None:
        flush(cur)
    return out


# --- flow signal: extreme versus the PAST-ONLY imbalance distribution --------------------------------


def flow_sides(timestamps, imbalance, signal, flow_pct, min_history=DEFAULT_MIN_HISTORY):
    """Per-bar entry side (+1 / -1 / ``None``) from the taker-imbalance extreme at each bar, judged past-only.

    The per-bar imbalance is the scalar; a high-buy extreme versus its own past is a long for flow_momentum /
    short for flow_contrarian, and vice versa for a high-sell extreme. The past-only quantile discipline and
    None-handling live in ``past_only_extreme_sides``; this only maps the two signal names to the invert flag
    (momentum rides the flow, contrarian fades it)."""
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    return past_only_extreme_sides(
        imbalance, invert=(signal == "flow_contrarian"), pct=float(flow_pct), min_history=int(min_history)
    )


def positions_from_bars(timestamps, closes, imbalance, cfg, start_index=0):
    """Decision-bar position vector for the flow signal. Positions depend ONLY on imbalance + timestamps
    (never on ``closes`` — passed solely so the signature mirrors the intraday builder and a caller can hold
    the aligned price for the backtest). The per-bar side is fed through intraday's hold-then-flat builder, so
    a fire at bar t occupies t+1..t+hold_bars, new fires inside the hold window are ignored, and entries before
    ``start_index`` are suppressed (the formation lookback warms the imbalance distribution)."""
    n = len(timestamps)
    if n == 0:
        return np.zeros(0)
    sides = flow_sides(
        timestamps,
        imbalance,
        str(cfg.get("signal", DEFAULT_SIGNAL)),
        float(cfg.get("flow_pct", DEFAULT_FLOW_PCT)),
        int(cfg.get("min_history", DEFAULT_MIN_HISTORY)),
    )
    return _build_positions(n, lambda t: sides[t], int(cfg.get("hold_bars", DEFAULT_HOLD_BARS)), start_index)


# --- data loading -------------------------------------------------------------------------------------


def _read_rows(path):
    with open(path) as fh:
        return json.load(fh)


def _load_flow_bars(asset, pairs, bar_minutes):
    """Decision bars (with imbalance) for ``asset`` over the requested (year, month) pairs, aggregated per file
    and concatenated. Missing months are skipped rather than faked; per-file aggregation keeps memory bounded
    (bar arrays, not the raw 1m rows) and is exact for day-dividing bar widths."""
    inst = data_catalog.instrument(asset)
    directory = inst.directory if inst else "binance"
    out = {"timestamp": [], "close": [], "volume": [], "taker_buy": [], "imbalance": []}
    for (y, m) in pairs:
        path = f"{directory}/{asset}-1m-{y}-{m}.json"
        if not os.path.exists(path):
            continue
        bars = flow_bars(_read_rows(path), bar_minutes)
        for key in out:
            out[key].extend(bars[key])
    return out


# --- the run contract ---------------------------------------------------------------------------------


def run(cfg):
    """Run one order-flow cell and return a trainer-contract RunSummary.

    The month before the test window is loaded as a formation lookback to warm the past-only imbalance
    distribution; only the TEST span is accounted, exactly as the daily and intraday lines report. The
    objective is the per-step OOS Sharpe; return_vs_hold_pct is versus a fee-charged buy-and-hold of the same
    asset over the same span. flow_coverage (fraction of accounted bars carrying a defined imbalance) is emitted
    so a broken aggregation surfaces as a metric instead of a silent all-flat book."""
    asset = str(cfg.get("asset", DEFAULT_ASSET))
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    bar_minutes = int(cfg.get("bar", DEFAULT_BAR))
    fee = float(cfg.get("transaction_fee", DEFAULT_FEE))

    _, test_pairs, meta = resolve_walk_forward_window(cfg)
    load_pairs = [_prior_month(test_pairs[0])] + list(test_pairs)
    bars = _load_flow_bars(asset, load_pairs, bar_minutes)
    closes = bars["close"]
    timestamps = bars["timestamp"]
    imbalance = bars["imbalance"]

    test_from_ms = _month_start_ms(meta["test_from"])
    start_index = next((i for i, ts in enumerate(timestamps) if ts >= test_from_ms), len(timestamps))

    pos = positions_from_bars(timestamps, closes, imbalance, cfg, start_index=start_index)
    closes_t = closes[start_index:]
    pos_t = pos[start_index:]
    ts_t = timestamps[start_index:]
    imb_t = imbalance[start_index:]

    equity = backtest(closes_t, pos_t, fee)
    n_rt = round_trips(pos_t)
    total_return_pct = (equity[-1] - 1.0) * 100.0 if len(equity) >= 1 else 0.0
    turnover = float(np.abs(np.diff(np.concatenate([[0.0], np.asarray(pos_t, dtype=float)]))).sum())
    covered = sum(1 for x in imb_t if _is_num(x))

    metrics = {
        "total_return_pct": _finite(total_return_pct),
        "baseline": 0.0,
        "n_trades": n_rt,
        "realized_cost_bps": _finite(turnover * fee * 10000),
        "final_net_worth": _finite(equity[-1]) if equity else 1.0,
        "trades_per_day": trades_per_day(n_rt, ts_t[0], ts_t[-1]) if len(ts_t) >= 2 else 0.0,
        "flow_coverage": _finite(covered / len(imb_t)) if imb_t else 0.0,
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
    parser = argparse.ArgumentParser(description="Order-flow (taker-imbalance) conditional-edge backtest")
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
        f"flow_coverage={m.get('flow_coverage', 0.0):.3f} "
        f"return_vs_hold_pct={m.get('return_vs_hold_pct', 0.0):.4f} -> {args.summary_out}"
    )


if __name__ == "__main__":
    main()
