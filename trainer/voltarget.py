"""Volatility-target screen (B1 §2) — the CHEAP falsifier for the second quantity.

Measured over the 12 symbols on disk, the trailing 20d RETURN carries almost nothing about the next 20d
return (corr +0.015) while the trailing 20d VOL carries a lot about the next 20d vol (corr +0.426, positive
on 12/12). Six consecutive nulls in this campaign all bet on the first quantity; nothing has ever traded the
second. This screen trades ONLY the second: it never forms a view on direction, it re-sizes a long book so
that its risk contribution is roughly constant, and it is therefore a bet on the persistence of variance
rather than the persistence of drift.

It is a NEW ENTRY POINT rather than an env change on purpose. `_position_size()`
(src/environment/base_crypto_env.py) is consulted only at ENTRY (trade_all_crypto_env.py), so a held position
is never re-sized and the env physically cannot express a continuously vol-targeted book. Rewiring it is a
3-4 week build; this screen is the cheap experiment that decides whether that spend is justified, under the
same governance the cross-sectional screen used: deliberately crude, published, deterministic, no training,
its own recordType so its cells never mix with the RL run store, and the SAME metric vocabulary (summary.py's
equity-curve statistics reused verbatim) so every existing gate, lens and scorecard reads it unchanged.

The benchmark is buy-and-hold the SAME asset, fee-netted the same way. Because the rule DE-RISKS it will
usually earn less raw return than buy-and-hold, so `return_vs_hold_pct` is the wrong lens for it; the gate
reads `sharpe_vs_hold` and `drawdown_vs_hold_pct`, both signed so that better-than-hold is POSITIVE.

Correctness properties pinned in test_voltarget.py. A vol-targeting rule is trivially easy to write with
lookahead — the moment the sizing target is a full-sample constant the backtest knows the future volatility
regime and every number it emits is fiction — so the strictly-past expanding target is the single most
important property in this module and is tested before the screen is allowed to produce a number.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from trainer import data_catalog
from trainer.summary import (
    _capture_stats,
    _finite,
    _max_drawdown_pct,
    _oos_stats,
    _provenance_fingerprint,
)
from trainer.walk_forward import resolve_walk_forward_window


def step_returns(prices):
    """Per-step return of the asset against its OWN previous bar. The first bar has no return (NaN), which
    is what lets the estimators below warm up honestly instead of counting a fabricated zero."""
    if prices.empty:
        return prices
    return prices / prices.shift(1) - 1.0


def trailing_vol(prices, window):
    """Realized vol at each bar: the population stdev of the last `window` per-step returns, using returns up
    to and INCLUDING that bar. NaN until `window` returns exist — the warm-up is a hole, never a zero,
    because a zero here would read as "no risk" and demand maximum size. Left un-annualized: the sizing rule
    is a ratio of two vols measured the same way, so any annualization factor cancels."""
    if prices.empty:
        return prices
    return step_returns(prices).rolling(int(max(1, window))).std(ddof=0)


def expanding_vol_target(vols):
    """The causal sizing target tau, indexed by the bar the DECISION is taken on: the running median of every
    trailing vol observed through that bar. The weight it sizes is traded into the NEXT bar, so relative to
    the return that weight is applied to (t-1 -> t) every input is strictly in the past.

    A median rather than a mean because one crisis bar must not permanently redefine "normal risk", and
    expanding rather than full-sample because a full-sample constant is lookahead: it would tell 2018 what
    2020 volatility looked like, which is the difference between a screen and a fiction."""
    if vols.empty:
        return vols
    return vols.expanding(min_periods=1).median()


def size_weight(target, vol, weight_cap):
    """The published rule: w = clip(tau / v_prev, 0, weight_cap). Returns None when either input is not yet
    knowable, so the caller HOLDS its previous book rather than writing a NaN into the weight series.

    The cap is what makes the rule safe rather than merely appealing: as realized vol collapses toward zero
    the raw ratio diverges, and an uncapped vol target is a machine for taking unbounded leverage in the
    quietest tape — historically the exact moment before it stops being quiet."""
    cap = max(0.0, float(weight_cap))
    if not np.isfinite(target) or not np.isfinite(vol):
        return None
    if vol <= 0.0:
        return cap
    return float(min(max(float(target) / float(vol), 0.0), cap))


def build_weights(prices, vol_window, weight_cap, rebalance_days, start=None):
    """The book HELD INTO each bar. Row t carries the weight decided at t-1 from vols observed through t-1,
    so a signal is never traded on its own bar and the weight applied to the t-1 -> t return is knowable at
    t-1. Rebalance every `rebalance_days` bars and hold the weight constant in between: turnover is the term
    that killed several earlier arms of this campaign, so the cost control is part of the rule rather than a
    post-hoc filter. `start` gates when DECISIONS may begin (the test window) while the estimators keep their
    full formation history."""
    weights = pd.Series(0.0, index=prices.index)
    if prices.empty:
        return weights
    vols = trailing_vol(prices, vol_window)
    target = expanding_vol_target(vols)
    step = max(1, int(rebalance_days))
    held = 0.0
    since = step  # decide on the first eligible bar
    for i, ts in enumerate(prices.index):
        # The book decided on PRIOR bars is what is held into this one.
        weights.iloc[i] = held
        if start is not None and ts < start:
            continue
        since += 1
        if since < step:
            continue
        nxt = size_weight(target.iloc[i], vols.iloc[i], weight_cap)
        if nxt is None:  # still warming up — stay flat rather than emit a NaN weight
            continue
        since = 0
        held = nxt
    return weights


def turnover_steps(weights):
    """Per-bar turnover of the book: |w_t - w_(t-1)|, with row 0 measured against CASH so the trade that OPENS
    the book is never free. `backtest` charges exactly this and `run` reports its sum, from this one
    definition, so the cost actually paid and the cost reported cannot drift apart — a bare `diff()` skips row
    0 and therefore under-reports precisely when a caller hands the screen a pre-loaded book."""
    if len(weights) == 0:
        return pd.Series(dtype=float)
    return (weights - weights.shift(1).fillna(0.0)).abs()


def backtest(prices, weights, fee):
    """Equity curve of the sized book, charging `fee` on TURNOVER (both the increase and the decrease of the
    weight, and the opening trade into row 0's book). Row t's weight multiplies the t-1 -> t return, which is
    the whole causality contract of the screen expressed in the accounting.

    Charging row 0 is what makes the benchmark comparison an identity rather than an assertion: a book that is
    always fully invested reproduces `hold_curve` bar for bar, so `sharpe_vs_hold` and `drawdown_vs_hold_pct`
    cannot be tilted by a fee landing in a different bar on one side than the other. Under the `start` gate
    the screen's own book is flat at row 0, so no cell's number changes; the rule holds anyway because
    `backtest` is what any future caller will reach for."""
    if prices.empty:
        return pd.Series(dtype=float)
    rets = step_returns(prices).fillna(0.0)
    turns = turnover_steps(weights)
    equity = [1.0 * (1.0 - fee * float(turns.iloc[0]))]
    for i in range(1, len(prices.index)):
        gross = float(weights.iloc[i]) * float(rets.iloc[i])
        equity.append(equity[-1] * (1.0 + gross) * (1.0 - fee * float(turns.iloc[i])))
    return pd.Series(equity, index=prices.index)


def hold_curve(prices, fee):
    """The benchmark: buy-and-hold the SAME asset, charged one entry's worth of fee so both curves are netted
    the same way. Every `*_vs_hold` metric is measured against this."""
    if prices.empty:
        return pd.Series(dtype=float)
    rets = step_returns(prices).fillna(0.0)
    equity = [1.0 * (1.0 - fee)]
    for i in range(1, len(prices.index)):
        equity.append(equity[-1] * (1.0 + float(rets.iloc[i])))
    return pd.Series(equity, index=prices.index)


def benchmark_metrics(equity, hold):
    """The comparison lens this screen is actually gated on, built by pushing the HOLD curve through the same
    `_oos_stats` / `_max_drawdown_pct` helpers as the strategy curve.

    A de-risking rule earns less raw return than buy-and-hold almost by definition, so `return_vs_hold_pct`
    would reject it for working as intended. `sharpe_vs_hold` and `drawdown_vs_hold_pct` are the honest reads,
    and both are signed so that better-than-hold is POSITIVE — drawdowns are negative percentages, so a
    shallower one minus a deeper one is a positive number. Empty (skippable via metrics.update) when either
    curve is too short for the underlying helpers."""
    strat, bench = _oos_stats(equity), _oos_stats(hold)
    strat_dd, bench_dd = _max_drawdown_pct(equity), _max_drawdown_pct(hold)
    if not strat or not bench or not strat_dd or not bench_dd:
        return {}
    return {
        "oos_sharpe": strat["oos_sharpe"],
        "hold_sharpe": bench["oos_sharpe"],
        "sharpe_vs_hold": _finite(strat["oos_sharpe"] - bench["oos_sharpe"]),
        "hold_max_drawdown_pct": bench_dd["max_drawdown_pct"],
        "drawdown_vs_hold_pct": _finite(strat_dd["max_drawdown_pct"] - bench_dd["max_drawdown_pct"]),
    }


def _load_asset(symbol, pairs):
    """Daily bars for one symbol over the requested (year, month) pairs, on the asset's own clock. Bars with
    no price are dropped rather than forward-filled: a hole must not become a flat return that quietly lowers
    realized vol and therefore raises the weight."""
    inst = data_catalog.instrument(symbol)
    directory = inst.directory if inst else "binance"
    paths = [f"{directory}/{symbol}-1d-{y}-{m}.json" for (y, m) in pairs]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return pd.Series(dtype=float)
    df = pd.concat([pd.read_json(p) for p in paths], ignore_index=True)
    if "timestamp_close" not in df.columns or "price" not in df.columns:
        return pd.Series(dtype=float)
    s = pd.Series(
        pd.to_numeric(df["price"], errors="coerce").values,
        index=pd.to_datetime(df["timestamp_close"]),
    )
    return s[~s.index.duplicated(keep="last")].sort_index().dropna()


def run(cfg):
    """Run one screen cell and return a trainer-contract RunSummary."""
    asset = str(cfg.get("asset", "SPY"))
    vol_window = int(cfg.get("vol_window", 20))
    weight_cap = float(cfg.get("weight_cap", 1.0))
    rebalance_days = int(cfg.get("rebalance_days", 5))
    fee = float(cfg.get("transaction_fee", 0.0002))

    train_pairs, test_pairs, window = resolve_walk_forward_window(cfg)
    prices = _load_asset(asset, list(train_pairs) + list(test_pairs))
    if prices.empty:
        raise SystemExit(f"no 1d bars on disk for {asset!r} over {window.get('walk_forward_window')}")
    # The vol estimator and its expanding target may look back into the TRAIN span, but only the TEST window
    # is accounted — the same out-of-sample rule the single-asset line reports under.
    test_start = pd.to_datetime(f"{test_pairs[0][0]}-{test_pairs[0][1]:02d}-01")
    weights = build_weights(prices, vol_window, weight_cap, rebalance_days, start=test_start)
    oos = prices.index >= test_start
    prices_oos = prices[oos]
    if len(prices_oos) < 2:
        raise SystemExit(f"{asset} has {len(prices_oos)} test bars in {window.get('walk_forward_window')}")
    weights_oos = weights[oos]
    equity = backtest(prices_oos, weights_oos, fee)
    hold = hold_curve(prices_oos, fee)

    total_return = float(equity.iloc[-1] - 1.0) * 100
    hold_return = float(hold.iloc[-1] - 1.0) * 100
    turns = turnover_steps(weights_oos)
    turnover = float(turns.sum())
    n_rebalances = int((turns > 1e-12).sum())
    metrics = {
        "total_return_pct": _finite(total_return),
        "hold_return_pct": _finite(hold_return),
        "return_vs_hold_pct": _finite(total_return - hold_return),
        "n_trades": n_rebalances,
        "turnover": _finite(turnover),
        "realized_cost_bps": _finite(turnover * fee * 10000),
        "mean_exposure": _finite(float(weights_oos.mean())),
        "max_exposure": _finite(float(weights_oos.max())),
        "bars": int(len(prices_oos)),
    }
    metrics.update(_oos_stats(list(equity.values)))
    metrics.update(_max_drawdown_pct(list(equity.values)))
    metrics.update(_capture_stats(list(equity.values), list(prices_oos.values)))
    metrics.update(benchmark_metrics(list(equity.values), list(hold.values)))
    summary = {
        "objective": _finite(metrics.get("oos_sharpe", 0.0)),
        "metrics": metrics,
        "dataset": {
            "asset": asset,
            "timeframe": "1d",
            "candles": int(len(prices_oos)),
            "from": str(prices_oos.index[0]),
            "to": str(prices_oos.index[-1]),
        },
        "health": {"status": "ok", "flags": []},
        "config": dict(cfg),
        "walk_forward_window": window.get("walk_forward_window"),
    }
    try:
        summary["provenance"] = _provenance_fingerprint(cfg, dict(cfg))
    except Exception:
        pass
    return summary


def main():
    parser = argparse.ArgumentParser(description="Volatility-target screen (B1 §2)")
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    with open(args.config_json) as fh:
        cfg = json.load(fh)
    summary = run(cfg)
    with open(args.summary_out, "w") as fh:
        json.dump(summary, fh)
    print(f"objective(oos_sharpe)={summary['objective']:.4f} status=ok -> {args.summary_out}")


if __name__ == "__main__":
    main()
