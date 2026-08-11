"""World-model probe — a DRIVER-CONDITIONED directional timing model (the B4 "world model" line, generalised).

The earlier B4 macro-regime overlay timed CRYPTO exposure from a few macro trends and nulled. This asks the same
question the "scrutinise a company / commodity and model what moves it" idea poses, but with the program's
discipline: read an asset's economically-motivated FUNDAMENTAL DRIVERS point-in-time and take a long / short /
flat position from their past-only trend — does conditioning on the drivers beat cost, DSR-corrected, OOS?

The seed case is GOLD, whose price has three canonical, exogenous, slow macro drivers: the 10y real rate
(DFII10 — gold rises when real rates FALL), the broad dollar (DTWEXBGS — gold rises when the dollar FALLS), and
the 10y inflation breakeven (DGS10 - DFII10 — gold rises when expected inflation RISES). The driver set is
declared per asset (DRIVER_SETS), so copper, silver, oil etc. can each get their own world model without engine
changes. worldmodel is the thesis; worldmodel_inverse is the mirror control that must FAIL if the drivers carry
real directional content. Gold has a strong secular uptrend, so return_vs_hold is exposure-biased — the honest
decider is the model's own DSR-deflated oos_sharpe, with sharpe_vs_hold reported alongside.

Macro values are joined POINT-IN-TIME via pit_fusion (known only from their release instant, DST-aware, never
the reference period — the leakage game). The composite side decided at day t sets the position for day t+1;
positions never read price. Deterministic and model-free (the seed is contract-only).

Guards pinned in test_worldmodel.py: each driver scalar is past-only-trended (value[t] vs value[t-lookback],
both known by t); the breakeven spread reads TWO DISTINCT series (DGS10 - DFII10); the composite requires ALL
declared drivers defined; and the side at t sets pos[t+1], never pos[t].
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

SIGNALS = ("worldmodel", "worldmodel_inverse")
DEFAULT_SIGNAL = "worldmodel"

# A driver is a named, economically-signed macro trend: `series` are the macro ids, `combine` maps them to one
# scalar per bar ('level' = the single series; 'spread' = series[0] - series[1]), and a past-only move of that
# scalar in `bullish_on` ('falling'/'rising') over `lookback` votes the asset BULLISH.
_REAL_RATE = {"name": "real_rate", "series": ("DFII10",), "combine": "level", "bullish_on": "falling"}
_USD = {"name": "usd", "series": ("DTWEXBGS",), "combine": "level", "bullish_on": "falling"}
_BREAKEVEN_T10YIE = {"name": "breakeven", "series": ("T10YIE",), "combine": "level", "bullish_on": "rising"}

DRIVER_SETS = {
    # Original gold set — DFII10 is DOUBLE-COUNTED (real-rate leg AND one term of breakeven = DGS10-DFII10);
    # kept only to reproduce the first (INCONCLUSIVE) gold probe. Use gold_macro3b for the fair test.
    "gold_macro3": (
        _REAL_RATE, _USD,
        {"name": "breakeven", "series": ("DGS10", "DFII10"), "combine": "spread", "bullish_on": "rising"},
    ),
    # De-collinearised gold set: breakeven from the standalone FRED series T10YIE, so DFII10 appears ONCE.
    "gold_macro3b": (_REAL_RATE, _USD, _BREAKEVEN_T10YIE),
    # Silver is a precious metal driven by the same monetary channel as gold (real rate + USD + inflation
    # expectations), plus an industrial component the macro set does not capture.
    "silver_macro3": (_REAL_RATE, _USD, _BREAKEVEN_T10YIE),
    # Copper ("Dr. Copper") is INDUSTRIAL — its real drivers (China demand, LME inventories) are NOT in the mined
    # macro set. This is a FINANCIAL-CONDITIONS model only: weak dollar + easier real rate + a steepening curve
    # (reflation/growth) are bullish. Deliberately incomplete — an honest test of what financial macro alone sees.
    "copper_macro3": (
        _USD, _REAL_RATE,
        {"name": "growth_curve", "series": ("T10Y2Y",), "combine": "level", "bullish_on": "rising"},
    ),
    # Real-rate-ALONE gold timer: the single most-cited gold driver, isolated. Closes the "equal-weight majority
    # vote masked a single predictive driver" escape hatch — if THIS also nulls (incl. the 2020-21 favorable
    # regime), the composite null generalises to the real-rate channel.
    "gold_realrate1": (_REAL_RATE,),
}
DEFAULT_DRIVER_SET = "gold_macro3"

DEFAULT_ASSET = "GOLD"
DEFAULT_LOOKBACK = 63       # daily bars behind the trend comparison (~3 months)
DEFAULT_FEE = 0.0005        # per-side, realistic for a liquid gold future / ETF
BAR_MINUTES = 1440
MACRO_DIR = "macro"


def resolve_driver_set(name):
    if name not in DRIVER_SETS:
        raise SystemExit(f"unknown driver_set {name!r}; choose one of {sorted(DRIVER_SETS)}")
    return DRIVER_SETS[name]


# --- driver scalar + past-only vote ------------------------------------------------------------------


def driver_scalar(driver, values_by_series):
    """Per-bar scalar for one driver: the single series' value ('level'), or the difference of two DISTINCT
    series ('spread', series[0] - series[1]). ``None`` at any bar where a needed leg is ``None``."""
    combine = driver["combine"]
    if combine == "level":
        return list(values_by_series[driver["series"][0]])
    if combine == "spread":
        a = values_by_series[driver["series"][0]]
        b = values_by_series[driver["series"][1]]
        return [(_x - _y) if (_is_num(_x) and _is_num(_y)) else None for _x, _y in zip(a, b)]
    raise SystemExit(f"unknown combine {combine!r}")


def driver_bullish(driver, values_by_series, lookback):
    """Per-bar bullish flag (True/False/``None``) for one driver: True when its scalar has moved in the
    ``bullish_on`` direction ('falling'/'rising') over ``lookback`` bars, comparing scalar[t] against
    scalar[t-lookback] (both past). ``None`` until a value ``lookback`` bars back exists, or where undefined."""
    s = driver_scalar(driver, values_by_series)
    falling = driver["bullish_on"] == "falling"
    lb = int(lookback)
    out = [None] * len(s)
    for t in range(len(s)):
        if t - lb >= 0 and _is_num(s[t]) and _is_num(s[t - lb]):
            out[t] = (s[t] <= s[t - lb]) if falling else (s[t] >= s[t - lb])
    return out


def worldmodel_sides(values_by_series, driver_set, lookback):
    """Per-bar composite side (+1 long / -1 short / 0 flat / ``None`` warm-up). Every declared driver must be
    defined at t (else ``None``); the side is the majority vote of the drivers' bullish flags — long if more
    bullish than bearish, short if more bearish, flat on a tie."""
    votes = [driver_bullish(d, values_by_series, lookback) for d in driver_set]
    n = len(votes[0]) if votes else 0
    out = [None] * n
    for t in range(n):
        col = [v[t] for v in votes]
        if any(x is None for x in col):
            continue
        bull = sum(1 for x in col if x)
        bear = len(col) - bull
        out[t] = 1 if bull > bear else (-1 if bear > bull else 0)
    return out


def positions_from_sides(timestamps, sides, signal, start_index=0):
    """Long/short/flat exposure vector: the composite side decided at day t sets exposure for day t+1
    (+1 long / -1 short / 0 flat). worldmodel_inverse flips the sign (the control). A ``None`` (warm-up) side is
    flat, and exposure before ``start_index`` is suppressed."""
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    n = len(sides)
    pos = np.zeros(n)
    flip = -1.0 if signal == "worldmodel_inverse" else 1.0
    for t in range(n - 1):
        s = sides[t]
        if s is None:
            continue
        if t + 1 >= start_index:
            pos[t + 1] = flip * float(s)
    return pos


# --- point-in-time macro join ------------------------------------------------------------------------


def _macro_asof_values(series_id, timestamps):
    """The point-in-time as-of value of ``series_id`` at each bar (latest RELEASED by it, via pit_fusion —
    never a future revision or the reference period). All ``None`` if the series is not mined."""
    path = f"{MACRO_DIR}/{series_id}.json"
    if not os.path.exists(path):
        return [None] * len(timestamps)
    with open(path) as fh:
        obs = json.load(fh)
    return fuse_series(list(timestamps), obs, publish_time_for(series_id))


def composite_sides(timestamps, cfg):
    """Per-bar composite side for the cfg's driver_set + lookback, loading each needed macro series' as-of value
    aligned to ``timestamps``."""
    driver_set = resolve_driver_set(str(cfg.get("driver_set", DEFAULT_DRIVER_SET)))
    needed = {s for d in driver_set for s in d["series"]}
    values = {s: _macro_asof_values(s, timestamps) for s in needed}
    return worldmodel_sides(values, driver_set, int(cfg.get("lookback", DEFAULT_LOOKBACK)))


# --- the run contract ---------------------------------------------------------------------------------


def run(cfg):
    """Run one world-model cell and return a trainer-contract RunSummary. The full training span is loaded so the
    trend + as-of macro are warm at the test start; only the TEST span is accounted. The objective is the
    per-step OOS Sharpe; sharpe_vs_hold and return_vs_hold are reported against a fee-charged buy-and-hold."""
    asset = str(cfg.get("asset", DEFAULT_ASSET))
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    resolve_driver_set(str(cfg.get("driver_set", DEFAULT_DRIVER_SET)))
    fee = float(cfg.get("transaction_fee", DEFAULT_FEE))

    train_pairs, test_pairs, meta = resolve_walk_forward_window(cfg)
    load_pairs = list(train_pairs) + list(test_pairs)
    bars = _load_bars(asset, load_pairs, BAR_MINUTES)
    closes = bars["close"]
    timestamps = bars["timestamp"]

    test_from_ms = _month_start_ms(meta["test_from"])
    start_index = next((i for i, ts in enumerate(timestamps) if ts >= test_from_ms), len(timestamps))

    sides = composite_sides(timestamps, cfg)
    pos = positions_from_sides(timestamps, sides, signal, start_index=start_index)
    closes_t = closes[start_index:]
    pos_t = pos[start_index:]
    ts_t = timestamps[start_index:]

    equity = backtest(closes_t, pos_t, fee)
    n_rt = round_trips(pos_t)
    total_return_pct = (equity[-1] - 1.0) * 100.0 if len(equity) >= 1 else 0.0
    turnover = float(np.abs(np.diff(np.concatenate([[0.0], np.asarray(pos_t, dtype=float)]))).sum())
    time_in_market = 100.0 * float(np.mean([1.0 if p != 0.0 else 0.0 for p in pos_t])) if len(pos_t) else 0.0
    long_share = 100.0 * float(np.mean([1.0 if p > 0 else 0.0 for p in pos_t])) if len(pos_t) else 0.0

    metrics = {
        "total_return_pct": _finite(total_return_pct),
        "baseline": 0.0,
        "n_trades": n_rt,
        "n_switches": int(round(turnover)),
        "time_in_market_pct": _finite(time_in_market),
        "long_share_pct": _finite(long_share),
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
    parser = argparse.ArgumentParser(description="World-model driver-conditioned directional timing")
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
