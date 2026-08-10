"""Macro-regime probe — a long-or-flat TIMING overlay (the B4 "world-model" line).

Every prior line sought a per-trade directional edge and found none — public, price-derived, scheduled and
mechanical information is all priced in. This asks a different question: can a SLOW macro regime estimate
(rates easing/tightening, the yield curve, the jobless-claims trend) time crypto EXPOSURE — hold in risk-on,
sit in cash in risk-off — and beat buy-and-hold by dodging drawdowns? It is an allocation overlay, not a
per-trade signal, so the honest test is beating hold across a BULL and a BEAR window at once: a real regime
signal must stay invested in the bull AND go flat in the bear. Crypto is a macro-sensitive risk asset and
macro regimes are persistent (months), so the overlay is low-turnover — cost is not the enemy, the signal's
skill is.

Macro values are joined POINT-IN-TIME via pit_fusion (known only from their release instant, DST-aware, never
the reference period) — the whole leakage game here. The overlay reuses the intraday backtest (a position of
1.0 = held, 0.0 = cash; the switch fee is charged on turnover) so every existing gate, lens and scorecard reads
a regime cell unchanged. regime_overlay is the thesis; regime_inverse (hold in risk-OFF) is the control that
should FAIL if the signal has real directional content. Deterministic and model-free (the seed is contract-only).

Guards pinned in test_regime.py: the rule is PAST-ONLY (the as-of value at t vs the as-of value `lookback` days
earlier, both known by t); the macro value at a day is the latest RELEASED by it; and the regime decided at day
t sets exposure for day t+1's return, never day t's own.
"""

import argparse
import json
import os

import numpy as np

from trainer.intraday import (
    _is_num,
    _iso_from_ms,
    _load_bars,
    _month_start_ms,
    backtest,
    round_trips,
)
from trainer.pit_fusion import fuse_series, publish_time_for
from trainer.summary import (
    _finite,
    _hold_equity,
    _max_drawdown_pct,
    _oos_stats,
    _provenance_fingerprint,
)
from trainer.walk_forward import resolve_walk_forward_window

# regime_overlay holds crypto in risk-ON and sits in cash in risk-OFF (the thesis). regime_inverse is the
# mirror control (hold in risk-OFF), which should lose if the macro signal carries real directional content.
SIGNALS = ("regime_overlay", "regime_inverse")
DEFAULT_SIGNAL = "regime_overlay"

# Each rule reads a small, economically-motivated set of RELEASED macro series (never price) and votes risk-on
# from a past-only trend. rates_easing: the 10Y yield has fallen over the lookback (easing -> risk-on).
# curve_steepening: the 10Y-2Y spread has risen (steepening out of inversion -> risk-on). claims_falling:
# initial jobless claims have fallen (labour firming -> risk-on). risk_composite: >=2 of the three.
RULE_SERIES = {
    "rates_easing": ("DGS10",),
    "curve_steepening": ("T10Y2Y",),
    "claims_falling": ("ICSA",),
    "risk_composite": ("DGS10", "T10Y2Y", "ICSA"),
}

DEFAULT_ASSET = "BTCUSDT"
DEFAULT_RULE = "risk_composite"
DEFAULT_LOOKBACK = 63       # daily bars behind the trend comparison (~3 months)
DEFAULT_FEE = 0.001
BAR_MINUTES = 1440          # daily decisions — the cadence a macro overlay operates at
MACRO_DIR = "macro"
_DAY_MS = 86_400_000


# --- the regime rule: past-only macro trend ----------------------------------------------------------


def _trend_on(values, lookback, direction):
    """Per-bar risk-on flag from a single series' trend: True when the as-of value has moved in the risk-on
    ``direction`` ('falling' or 'rising') over ``lookback`` bars, comparing value[t] against value[t-lookback]
    (both past). ``None`` until a value ``lookback`` bars back exists or where either endpoint is undefined."""
    n = len(values)
    out = [None] * n
    lb = int(lookback)
    for t in range(n):
        if t - lb >= 0 and _is_num(values[t]) and _is_num(values[t - lb]):
            out[t] = (values[t] <= values[t - lb]) if direction == "falling" else (values[t] >= values[t - lb])
    return out


def apply_rule(values_by_series, rule, lookback):
    """Per-bar risk-on flag (bool / ``None``) for a named regime ``rule`` over past-only macro trends.

    Single-series rules are a trailing trend in the risk-on direction; ``risk_composite`` is risk-on when at
    least two of {rates easing, curve steepening, claims falling} agree (all three must be defined, else
    ``None``). An unknown rule is refused rather than silently trading nothing."""
    if rule not in RULE_SERIES:
        raise SystemExit(f"unknown rule {rule!r}; choose one of {sorted(RULE_SERIES)}")
    if rule == "rates_easing":
        return _trend_on(values_by_series["DGS10"], lookback, "falling")
    if rule == "curve_steepening":
        return _trend_on(values_by_series["T10Y2Y"], lookback, "rising")
    if rule == "claims_falling":
        return _trend_on(values_by_series["ICSA"], lookback, "falling")
    a = _trend_on(values_by_series["DGS10"], lookback, "falling")
    b = _trend_on(values_by_series["T10Y2Y"], lookback, "rising")
    c = _trend_on(values_by_series["ICSA"], lookback, "falling")
    n = len(a)
    out = [None] * n
    for t in range(n):
        votes = [x[t] for x in (a, b, c) if x[t] is not None]
        if len(votes) == 3:
            out[t] = sum(votes) >= 2
    return out


def _macro_asof_values(series_id, timestamps):
    """The point-in-time as-of value of ``series_id`` at each bar timestamp (the latest value RELEASED at or
    before that bar, via pit_fusion — never a future revision or the reference period). ``None`` before the
    first release; an un-mined series yields all ``None`` rather than raising."""
    path = f"{MACRO_DIR}/{series_id}.json"
    if not os.path.exists(path):
        return [None] * len(timestamps)
    with open(path) as fh:
        obs = json.load(fh)
    return fuse_series(list(timestamps), obs, publish_time_for(series_id))


def regime_on(timestamps, cfg):
    """Per-bar risk-on flag from the cfg's ``rule`` and ``lookback``, loading each needed macro series' as-of
    value aligned to ``timestamps`` and applying the past-only rule."""
    rule = str(cfg.get("rule", DEFAULT_RULE))
    if rule not in RULE_SERIES:
        raise SystemExit(f"unknown rule {rule!r}; choose one of {sorted(RULE_SERIES)}")
    values = {s: _macro_asof_values(s, timestamps) for s in RULE_SERIES[rule]}
    return apply_rule(values, rule, int(cfg.get("lookback", DEFAULT_LOOKBACK)))


def regime_positions(timestamps, closes, cfg, start_index=0, risk_on=None):
    """Long-or-flat exposure vector: the regime decided at day t sets exposure for day t+1 (1.0 held / 0.0
    cash). regime_overlay holds in risk-on; regime_inverse holds in risk-off. A ``None`` (warm-up) regime is
    flat, and exposure before ``start_index`` is suppressed (the formation lookback warms the trend). ``risk_on``
    may be supplied directly (tests); otherwise it is computed from the macro."""
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    n = len(timestamps)
    if n == 0:
        return np.zeros(0)
    if risk_on is None:
        risk_on = regime_on(timestamps, cfg)
    inverse = signal == "regime_inverse"
    pos = np.zeros(n)
    for t in range(n - 1):
        ro = risk_on[t]
        if ro is None:
            continue
        exposure = 1.0 if bool(ro) else 0.0
        if inverse:
            exposure = 1.0 - exposure
        if t + 1 >= start_index:
            pos[t + 1] = exposure
    return pos


# --- the run contract ---------------------------------------------------------------------------------


def run(cfg):
    """Run one macro-regime cell and return a trainer-contract RunSummary.

    The full training span is loaded so the trend + as-of macro are warm at the test start; only the TEST span
    is accounted. The objective is the per-step OOS Sharpe; the headline is sharpe_vs_hold and the drawdown /
    time-in-market of the overlay versus a fee-charged buy-and-hold of the same asset over the same span."""
    asset = str(cfg.get("asset", DEFAULT_ASSET))
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    fee = float(cfg.get("transaction_fee", DEFAULT_FEE))

    train_pairs, test_pairs, meta = resolve_walk_forward_window(cfg)
    load_pairs = list(train_pairs) + list(test_pairs)
    bars = _load_bars(asset, load_pairs, BAR_MINUTES)
    closes = bars["close"]
    timestamps = bars["timestamp"]

    test_from_ms = _month_start_ms(meta["test_from"])
    start_index = next((i for i, ts in enumerate(timestamps) if ts >= test_from_ms), len(timestamps))

    pos = regime_positions(timestamps, closes, cfg, start_index=start_index)
    closes_t = closes[start_index:]
    pos_t = pos[start_index:]
    ts_t = timestamps[start_index:]

    equity = backtest(closes_t, pos_t, fee)
    n_rt = round_trips(pos_t)
    total_return_pct = (equity[-1] - 1.0) * 100.0 if len(equity) >= 1 else 0.0
    turnover = float(np.abs(np.diff(np.concatenate([[0.0], np.asarray(pos_t, dtype=float)]))).sum())
    time_in_market = 100.0 * float(np.mean([1.0 if p != 0.0 else 0.0 for p in pos_t])) if len(pos_t) else 0.0

    metrics = {
        "total_return_pct": _finite(total_return_pct),
        "baseline": 0.0,
        "n_trades": n_rt,
        "n_switches": int(round(turnover)),
        "time_in_market_pct": _finite(time_in_market),
        "realized_cost_bps": _finite(turnover * fee * 10000),
        "final_net_worth": _finite(equity[-1]) if equity else 1.0,
    }
    metrics.update(_oos_stats(equity))
    metrics.update(_max_drawdown_pct(equity))

    benchmark = {}
    prices = [p for p in closes_t if _is_num(p) and p > 0]
    if len(prices) >= 2:
        round_trip = (1.0 - fee) ** 2
        hold_return_pct = (prices[-1] / prices[0] * round_trip - 1.0) * 100.0
        benchmark["hold_return_pct"] = _finite(hold_return_pct)
        hold_equity = _hold_equity(prices, round_trip)
        hstats = _oos_stats(hold_equity)
        hold_sharpe = hstats.get("oos_sharpe")
        if hold_sharpe is not None:
            benchmark["hold_sharpe"] = hold_sharpe
            metrics["sharpe_vs_hold"] = _finite(metrics.get("oos_sharpe", 0.0) - hold_sharpe)
        hdd = _max_drawdown_pct(hold_equity)
        if "max_drawdown_pct" in hdd:
            benchmark["hold_max_drawdown_pct"] = hdd["max_drawdown_pct"]
        metrics["hold_return_pct"] = benchmark["hold_return_pct"]
        metrics["return_vs_hold_pct"] = _finite(total_return_pct - benchmark["hold_return_pct"])
        metrics["hold_net_of_fees"] = True
    metrics.setdefault("sharpe_vs_hold", 0.0)

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
    parser = argparse.ArgumentParser(description="Macro-regime long-or-flat timing overlay")
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
        f"sharpe_vs_hold={m.get('sharpe_vs_hold', 0.0):.4f} "
        f"time_in_market={m.get('time_in_market_pct', 0.0):.1f}% n_switches={m.get('n_switches', 0)} "
        f"return_vs_hold_pct={m.get('return_vs_hold_pct', 0.0):.4f} -> {args.summary_out}"
    )


if __name__ == "__main__":
    main()
