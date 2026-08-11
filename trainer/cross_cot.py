"""Cross-sectional COT probe — the structurally-different positioning thread (RELATIVE VALUE, market-neutral).

Every single-asset positioning probe nulled because its tempting cells were BETA (long a rising commodity). This
removes that beta by construction: each bar it ranks the basket by managed-money positioning extremity (the
trailing COT index of net/OI) and goes LONG the least-crowded commodities / SHORT the most-crowded
(cross_contrarian), or the reverse (cross_momentum) — a DOLLAR-NEUTRAL long/short across a metals+energy+ags
basket, so the common commodity beta cancels and only a genuine RELATIVE-positioning edge can survive. It reuses
cot.py's release-lagged join (`cot_values`) + trailing COT index (`cot_index_series`); the cross-sectional
ranking + the market-neutral portfolio backtest are the new surface. Deterministic and model-free.

Commodities trade on slightly different calendars, so they are aligned on a COMMON daily axis (the union of the
basket's bar dates) with each commodity's close forward-filled — the position decided at day t earns day t+1's
return, positions never read price, and the COT score is release-lagged + past-only.
"""

import argparse
import json

import numpy as np

from trainer import cot
from trainer.intraday import _iso_from_ms, _load_bars, _month_start_ms
from trainer.summary import (
    _finite,
    _max_drawdown_pct,
    _oos_stats,
    _provenance_fingerprint,
    _track_record_stats,
)
from trainer.walk_forward import resolve_walk_forward_window

SIGNALS = ("cross_contrarian", "cross_momentum")
DEFAULT_SIGNAL = "cross_contrarian"

BASKETS = {"complex6": ["GOLD", "SILVER", "COPPER", "WTI", "CORN", "WHEAT"]}
DEFAULT_BASKET = "complex6"

DEFAULT_K = 2                 # long the k least-crowded, short the k most-crowded
DEFAULT_INDEX_WINDOW = 756    # ~3yr trailing COT index (regime-adaptive)
DEFAULT_MIN_HISTORY = 126
DEFAULT_RELEASE_LAG = 4
DEFAULT_FEE = 0.0005
BAR_MINUTES = 1440


def cross_row_weights(scores, k, invert):
    """Market-neutral cross-sectional weights for one bar. ``scores`` = ``[(commodity, score|None)]``; go LONG
    the k LOWEST scores (+1/k) and SHORT the k HIGHEST (-1/k) — least-crowded long / most-crowded short
    (contrarian); ``invert`` flips the two (momentum). All-zero (no trade) when fewer than 2k commodities have a
    defined score. Weights sum to ~0 (dollar-neutral)."""
    weights = {c: 0.0 for c, _ in scores}
    defined = [(c, s) for c, s in scores if s is not None]
    if len(defined) < 2 * int(k):
        return weights
    ranked = sorted(defined, key=lambda cs: cs[1])
    long_w = (-1.0 if invert else 1.0) / float(k)
    short_w = (1.0 if invert else -1.0) / float(k)
    for c, _ in ranked[: int(k)]:
        weights[c] = long_w
    for c, _ in ranked[-int(k):]:
        weights[c] = short_w
    return weights


def _aligned(basket, load_pairs, release_lag, index_window, min_history):
    """Per-commodity, on a COMMON daily axis (union of the basket's bar timestamps): forward-filled close ->
    next-bar return, and the release-lagged trailing COT-index score. Returns (timestamps, {commodity: {ret, score}})."""
    closes_by, ts_set = {}, set()
    for c in basket:
        bars = _load_bars(c, load_pairs, BAR_MINUTES)
        m = {int(t): float(px) for t, px in zip(bars["timestamp"], bars["close"]) if px is not None and px > 0}
        closes_by[c] = m
        ts_set.update(m)
    timestamps = sorted(ts_set)
    out = {}
    for c in basket:
        m = closes_by[c]
        ff, last = [], None
        for t in timestamps:
            if t in m:
                last = m[t]
            ff.append(last)
        ret = [0.0] * len(timestamps)
        for i in range(1, len(timestamps)):
            if ff[i] is not None and ff[i - 1] is not None and ff[i - 1] > 0:
                ret[i] = ff[i] / ff[i - 1] - 1.0
        netoi = cot.cot_values(c, timestamps, release_lag)
        out[c] = {"ret": ret, "score": cot.cot_index_series(netoi, index_window, min_history)}
    return timestamps, out


def run(cfg):
    """Run one cross-sectional cell and return a trainer-contract RunSummary. The strategy is a dollar-neutral
    long/short across the basket, so there is no buy-and-hold benchmark; the decider is the DSR-deflated
    oos_sharpe of the long/short portfolio. Only the TEST span is accounted (the prior span warms the trailing
    index + the return series)."""
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    basket_name = str(cfg.get("basket", DEFAULT_BASKET))
    if basket_name not in BASKETS:
        raise SystemExit(f"unknown basket {basket_name!r}; choices: {sorted(BASKETS)}")
    basket = BASKETS[basket_name]
    k = int(cfg.get("cross_k", DEFAULT_K))
    fee = float(cfg.get("transaction_fee", DEFAULT_FEE))
    release_lag = int(cfg.get("release_lag_days", DEFAULT_RELEASE_LAG))
    index_window = int(cfg.get("cot_index_window", DEFAULT_INDEX_WINDOW))
    min_history = int(cfg.get("min_history", DEFAULT_MIN_HISTORY))
    invert = signal == "cross_momentum"

    train_pairs, test_pairs, meta = resolve_walk_forward_window(cfg)
    load_pairs = list(train_pairs) + list(test_pairs)
    timestamps, data = _aligned(basket, load_pairs, release_lag, index_window, min_history)
    n = len(timestamps)

    weights = {c: [0.0] * n for c in basket}
    for t in range(n):
        w = cross_row_weights([(c, data[c]["score"][t]) for c in basket], k, invert)
        for c in basket:
            weights[c][t] = w[c]

    test_from_ms = _month_start_ms(meta["test_from"])
    # portfolio return next-bar: the weights decided at t-1 earn t's return; turnover cost on weight change.
    port_ts, port_ret, exposure = [], [], []
    for t in range(1, n):
        gross = sum(weights[c][t - 1] * data[c]["ret"][t] for c in basket)
        turnover = sum(abs(weights[c][t - 1] - (weights[c][t - 2] if t >= 2 else 0.0)) for c in basket)
        port_ts.append(timestamps[t])
        port_ret.append(gross - fee * turnover)
        exposure.append(sum(abs(weights[c][t - 1]) for c in basket))

    keep = [i for i, ts in enumerate(port_ts) if ts >= test_from_ms]
    rets_t = [port_ret[i] for i in keep]
    exp_t = [exposure[i] for i in keep]
    ts_t = [port_ts[i] for i in keep]

    equity = list(np.cumprod([1.0 + r for r in rets_t])) if rets_t else []
    total_return_pct = (equity[-1] - 1.0) * 100.0 if equity else 0.0
    n_active = sum(1 for e in exp_t if e > 0)

    metrics = {
        "total_return_pct": _finite(total_return_pct),
        "baseline": 0.0,
        "n_trades": n_active,
        "gross_exposure": _finite(sum(exp_t) / len(exp_t)) if exp_t else 0.0,
        "realized_cost_bps": _finite(fee * (sum(exp_t) / len(exp_t) if exp_t else 0.0) * 2 * 10000),
        "final_net_worth": _finite(equity[-1]) if equity else 1.0,
        "signal_expectancy": _finite(sum(rets_t) / len(rets_t) * 100.0) if rets_t else 0.0,
        "signal_hit_rate": _finite(100.0 * sum(1 for r in rets_t if r > 0) / len(rets_t)) if rets_t else 0.0,
    }
    oos = _oos_stats(equity)
    metrics.update(oos)
    metrics.update(_track_record_stats(oos))
    metrics.update(_max_drawdown_pct(equity))

    objective = metrics.get("oos_sharpe", 0.0)
    stored_cfg = dict(cfg)
    summary = {
        "objective": _finite(objective),
        "metrics": metrics,
        "health": {"status": "ok" if n_active else "degenerate", "flags": [] if n_active else ["no_trades"]},
        "config": stored_cfg,
        "dataset": {
            "asset": basket_name,
            "timeframe": f"{BAR_MINUTES}m",
            "candles": len(rets_t),
            "walk_forward_window": meta["walk_forward_window"],
            "from": _iso_from_ms(ts_t[0]) if ts_t else None,
            "to": _iso_from_ms(ts_t[-1]) if ts_t else None,
        },
        "walk_forward_window": meta["walk_forward_window"],
    }
    try:
        summary["provenance"] = {"ranAt": cfg.get("ran_at"), **_provenance_fingerprint(cfg, stored_cfg)}
    except Exception:
        summary["provenance"] = {}
    if "seed" in cfg:
        summary["seed"] = int(cfg["seed"])
    return summary


def main():
    parser = argparse.ArgumentParser(description="Cross-sectional COT long/short backtest")
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
        f"n_active={m.get('n_trades', 0)} gross_exp={m.get('gross_exposure', 0.0):.2f} "
        f"signal_expectancy={m.get('signal_expectancy', 0.0):.4f} "
        f"total_return_pct={m.get('total_return_pct', 0.0):.4f} -> {args.summary_out}"
    )


if __name__ == "__main__":
    main()
