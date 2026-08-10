"""Liquidation-cascade probe — the mechanically-asymmetric line (detected, not scheduled).

A liquidation cascade is FORCED deleveraging: an exchange market-closes blown-up leveraged positions
regardless of price. A long cascade dumps aggressive SELLS into the book — price craters on a volume spike
with one-sided taker selling — and a short squeeze is the mirror. The bet is that this flow is
NON-informational (the seller has no choice), so the dislocation OVER-shoots and REVERTS: buy the forced-sell
crash, short the forced-buy squeeze. Unlike a scheduled release (priced in instantly) or contemporaneous flow,
a cascade is a mechanical asymmetry — the class the price/funding/order-flow/event nulls do NOT cover.

Real historical liquidation feeds are not freely available (binance deprecated the REST force-order endpoint;
the websocket stream is live-only), so the cascade is DETECTED from a proxy already on disk: a decision bar
whose (1) move clears k_move * trailing vol, (2) volume clears a past-only spike quantile, and (3) taker
imbalance is extreme ON THE SAME SIDE as the move — the forced aggressor flow that caused it. All three are
required; any one alone is just a volatile bar (and was already measured null by the vol-breakout and
order-flow probes). It reuses the order-flow bar aggregation (close + volume + signed imbalance) and the
intraday backtest mechanics, so every existing gate, lens and scorecard reads a cascade cell unchanged. The
thesis arm cascade_reversion FADES the cascade; the control cascade_momentum RIDES it. Deterministic and
model-free (the seed is contract-only).

Guards pinned in test_liquidation.py: the compound gate is a real AND (two of three never fires); the flow
must CONFIRM the move's side (a crash on aggressive buying is not a sell-cascade); every threshold is past-only;
and the cascade detected at bar t (from data known at its close) applies to bars t+1.. — the AFTERMATH — never
bar t's own dislocation return, so the pipeline is time-prefix causal.
"""

import argparse
import json

import numpy as np

from trainer.intraday import (
    _build_positions,
    _is_num,
    _iso_from_ms,
    _month_start_ms,
    _prior_month,
    _sign,
    _trade_edges,
    backtest,
    bar_returns,
    realized_vol,
    round_trips,
    trades_per_day,
)
from trainer.orderflow import _load_flow_bars
from trainer.summary import (
    _capture_stats,
    _finite,
    _hold_equity,
    _max_drawdown_pct,
    _oos_stats,
    _provenance_fingerprint,
)
from trainer.walk_forward import resolve_walk_forward_window

# fade the cascade (the forced-flow overshoot reverts) vs ride it. An unknown value is refused so a typo can
# never file one arm's result under the other's label.
SIGNALS = ("cascade_reversion", "cascade_momentum")
DEFAULT_SIGNAL = "cascade_reversion"

DEFAULT_ASSET = "BTCUSDT"
DEFAULT_BAR = 15          # cascades are fast; a tight decision bar catches the dislocation + its aftermath
DEFAULT_VOL_WINDOW = 20   # bars of trailing returns behind the realized-vol estimate
DEFAULT_K_MOVE = 3.0      # the move must clear k_move * trailing vol
DEFAULT_VOL_PCT = 0.90    # the volume must clear this quantile of the past-only volume distribution
DEFAULT_FLOW_PCT = 0.80   # the taker imbalance must be in this tail of the past-only imbalance distribution
DEFAULT_HOLD_BARS = 4
DEFAULT_MIN_HISTORY = 30
DEFAULT_FEE = 0.001


def cascade_sides(closes, volume, imbalance, cfg):
    """Per-bar entry side (+1 / -1 / ``None``) from the compound cascade detector.

    A bar t is a cascade when ALL of: |return_t| >= k_move * trailing_vol (through t-1); volume_t is at/above
    the ``vol_pct`` quantile of the volume observed strictly before t; and imbalance_t is in the ``flow_pct``
    tail ON THE SAME SIDE as the move (a down move needs aggressive selling, an up move aggressive buying) of
    the imbalance observed strictly before t. All three distributions are past-only. The cascade side is the
    sign of the move; cascade_reversion enters its negation (fade), cascade_momentum enters it (ride). A bar
    with an undefined return/vol, too little history, or any condition unmet is ``None``."""
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    fade = signal == "cascade_reversion"
    k_move = float(cfg.get("k_move", DEFAULT_K_MOVE))
    vol_pct = float(cfg.get("vol_pct", DEFAULT_VOL_PCT))
    flow_pct = float(cfg.get("flow_pct", DEFAULT_FLOW_PCT))
    min_history = int(cfg.get("min_history", DEFAULT_MIN_HISTORY))

    returns = bar_returns(closes)
    vol = realized_vol(returns, int(cfg.get("vol_window", DEFAULT_VOL_WINDOW)))
    n = len(closes)
    sides = [None] * n
    vol_hist = []  # volumes observed at bars strictly before the current one
    imb_hist = []  # imbalances observed at bars strictly before the current one
    for t in range(n):
        r = returns[t]
        v_prev = vol[t - 1] if t - 1 >= 0 else None
        volume_t = volume[t] if t < len(volume) else None
        imb_t = imbalance[t] if t < len(imbalance) else None
        # The move gate is cheap; the volume/imbalance quantiles are O(history) so they are computed ONLY on a
        # candidate move (cascades are rare) — gating here is a pure speedup, the compound AND is unchanged.
        if (
            _is_num(r) and _sign(r) != 0.0 and _is_num(v_prev)
            and _is_num(volume_t) and _is_num(imb_t)
            and len(vol_hist) >= min_history and len(imb_hist) >= min_history
            and abs(r) > k_move * v_prev
        ):
            volume_ok = volume_t >= float(np.quantile(vol_hist, vol_pct))
            if r < 0:
                flow_ok = imb_t <= float(np.quantile(imb_hist, 1.0 - flow_pct))
            else:
                flow_ok = imb_t >= float(np.quantile(imb_hist, flow_pct))
            if volume_ok and flow_ok:
                cascade_side = _sign(r)
                sides[t] = (-cascade_side if fade else cascade_side)
        if _is_num(volume_t):
            vol_hist.append(float(volume_t))
        if _is_num(imb_t):
            imb_hist.append(float(imb_t))
    return sides


def positions_from_bars(timestamps, closes, volume, imbalance, cfg, start_index=0):
    """Decision-bar position vector for the cascade signal. The per-bar cascade side is fed through intraday's
    hold-then-flat builder, so a cascade at bar t occupies t+1..t+hold_bars (never bar t's own dislocation
    return), a new cascade inside the hold window is ignored, and cascades before ``start_index`` are
    suppressed (the formation lookback warms the vol/volume/imbalance distributions)."""
    n = len(timestamps)
    if n == 0:
        return np.zeros(0)
    sides = cascade_sides(closes, volume, imbalance, cfg)
    return _build_positions(n, lambda t: sides[t], int(cfg.get("hold_bars", DEFAULT_HOLD_BARS)), start_index)


# --- the run contract ---------------------------------------------------------------------------------


def run(cfg):
    """Run one liquidation-cascade cell and return a trainer-contract RunSummary.

    The month before the test window is a formation lookback for the vol/volume/imbalance distributions; only
    the TEST span is accounted. The objective is the per-step OOS Sharpe; return_vs_hold_pct is versus a
    fee-charged buy-and-hold of the same asset. n_cascades (detections inside the accounted span) is emitted so
    a thin detection count is visible next to the result."""
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
    volume = bars["volume"]
    imbalance = bars["imbalance"]

    test_from_ms = _month_start_ms(meta["test_from"])
    start_index = next((i for i, ts in enumerate(timestamps) if ts >= test_from_ms), len(timestamps))

    sides = cascade_sides(closes, volume, imbalance, cfg)
    pos = _build_positions(len(timestamps), lambda t: sides[t], int(cfg.get("hold_bars", DEFAULT_HOLD_BARS)), start_index)
    closes_t = closes[start_index:]
    pos_t = pos[start_index:]
    ts_t = timestamps[start_index:]

    equity = backtest(closes_t, pos_t, fee)
    n_rt = round_trips(pos_t)
    total_return_pct = (equity[-1] - 1.0) * 100.0 if len(equity) >= 1 else 0.0
    turnover = float(np.abs(np.diff(np.concatenate([[0.0], np.asarray(pos_t, dtype=float)]))).sum())
    n_cascades = sum(1 for i in range(start_index, len(sides)) if sides[i] is not None)

    metrics = {
        "total_return_pct": _finite(total_return_pct),
        "baseline": 0.0,
        "n_trades": n_rt,
        "n_cascades": n_cascades,
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
    parser = argparse.ArgumentParser(description="Liquidation-cascade conditional-edge backtest")
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
        f"n_cascades={m.get('n_cascades', 0)} trades_per_day={m.get('trades_per_day', 0.0):.4f} "
        f"return_vs_hold_pct={m.get('return_vs_hold_pct', 0.0):.4f} -> {args.summary_out}"
    )


if __name__ == "__main__":
    main()
