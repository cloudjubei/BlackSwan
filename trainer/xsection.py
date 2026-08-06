"""Cross-sectional long/short screen (B1 §1) — the CHEAP falsifier for the fork.

Every single-asset formulation tried so far asks "will THIS market go up?", and across 264 measured cells
the answer carries no selectivity: the configs that beat buy-and-hold do it by being barely invested
(mean beta 0.19) with SYMMETRIC capture. This asks a different question — "will A outperform B?" — a
relative bet that is beta-neutral by construction.

Deliberately crude and deterministic (published cross-sectional momentum, no training, no RL): its job is
to KILL the 3-4 week multi-asset env build cheaply if the edge is not there. It emits the SAME metric
vocabulary as the single-asset line (reusing summary.py's equity-curve statistics verbatim) so every
existing gate, lens and scorecard reads it unchanged; the buy-and-hold benchmark is the equal-weight
universe basket rather than a single asset.

Correctness properties pinned in test_xsection.py, because a cross-sectional backtest fails in ways the
single-asset line cannot: a misaligned N-symbol join fabricates P&L, and ranking against today's symbol
list is survivorship bias.
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

# The screen's universe is PRE-REGISTERED (docs/cross-sectional-plan.md): choosing symbols after seeing
# results is selection-on-test. US-session instruments only — mixing 24/7 crypto onto a session calendar is
# the hardest alignment case and is deliberately deferred to §2 rather than fudged here.
UNIVERSES = {
    "macro": ["GOLD", "SPY", "UUP", "TLT", "IEF", "SHY"],
    "stocks": ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "JPM", "WMT"],
    "macro+stocks": [
        "GOLD", "SPY", "UUP", "TLT", "IEF", "SHY",
        "NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "JPM", "WMT",
    ],
}
DEFAULT_UNIVERSE = "macro+stocks"

# Which way the trailing-return rank is traded. "momentum" is the published rule the screen was built on;
# "reversal" is its exact inversion, added because the 96-cell momentum screen's worst window by far was
# stk-2023 at mean -63.50% — a loss that size is structure with the sign flipped, not noise. The two arms
# share every other line of this file so a reversal cell is directly comparable to its momentum twin.
SIGNALS = ("momentum", "reversal")
DEFAULT_SIGNAL = "momentum"


def align_prices(frames):
    """symbol -> frame(timestamp_close, price) onto ONE clock: a date-indexed price matrix, NaN where a
    symbol has no bar. A hole stays a hole — forward-filling it would invent a flat return and, worse,
    let a stale price win a cross-sectional rank."""
    if not frames:
        return pd.DataFrame()
    cols = {}
    for symbol, df in frames.items():
        s = pd.Series(
            pd.to_numeric(df["price"], errors="coerce").values,
            index=pd.to_datetime(df["timestamp_close"]),
        )
        cols[symbol] = s[~s.index.duplicated(keep="last")].sort_index()
    return pd.DataFrame(cols).sort_index()


def step_returns(prices):
    """Per-step return per symbol, measured against that symbol's OWN previous bar (so a gap in one
    symbol's calendar never pairs across another's). 0.0 where the symbol has no bar on either side."""
    if prices.empty:
        return prices
    out = {}
    for symbol in prices.columns:
        s = prices[symbol].dropna()
        out[symbol] = (s / s.shift(1) - 1.0).reindex(prices.index)
    return pd.DataFrame(out, index=prices.index).fillna(0.0)


def tradeable_mask(prices, lookback):
    """True where a symbol has a bar AND at least `lookback` PRIOR bars of its own. Enforces both the
    warm-up and the survivorship rule: a symbol that has not listed yet (or has too little history to rank)
    is invisible to the ranking, never a silently-zero row."""
    if prices.empty:
        return prices
    present = prices.notna()
    prior = present.cumsum().shift(1).fillna(0)
    return present & (prior >= int(lookback))


def momentum(prices, lookback):
    """Trailing `lookback`-bar return per symbol, over the symbol's OWN observations. Backward-looking
    only — the value at t uses bars at or before t."""
    if prices.empty:
        return prices
    out = {}
    for symbol in prices.columns:
        s = prices[symbol].dropna()
        out[symbol] = (s / s.shift(int(lookback)) - 1.0).reindex(prices.index)
    return pd.DataFrame(out, index=prices.index)


def build_weights(prices, lookback, k, rebalance_days, long_only, signal=DEFAULT_SIGNAL, start=None):
    """The book HELD INTO each bar. A rank computed from data up to t is applied to the NEXT step's return
    (row t carries what was decided at t-1), so a signal is never traded on its own bar. Rebalance every
    `rebalance_days` bars; between rebalances the book is held. Long/short is equal-weight both sides and
    therefore NET FLAT — the beta-neutrality the whole fork rests on.

    `signal` only chooses which END of the trailing-return rank is bought: "momentum" long the top k,
    "reversal" long the bottom k. It defaults to "momentum", so every book built before the lever existed
    is reproduced bar-for-bar and the 96 persisted screen cells stay comparable. An unrecognised value is
    an error rather than a fallback — a typo'd lever that quietly ranked as momentum would file a momentum
    result under a reversal label, which corrupts the evidence trail instead of merely losing money."""
    if signal not in SIGNALS:
        raise ValueError(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    if int(lookback) < 1:
        # A non-positive formation window inverts momentum()'s shift into a FORWARD ratio, so the rank is
        # computed from bars that have not happened, and tradeable_mask waves it through because
        # `prior >= lookback` is trivially true. The screen would report a clean-looking result built on a
        # peek at the future, which is the one failure mode this module exists to make impossible.
        raise ValueError(f"lookback must be >= 1; got {lookback!r}")
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if prices.empty:
        return weights
    mom = momentum(prices, lookback)
    ok = tradeable_mask(prices, lookback)
    k = max(1, int(k))
    step = max(1, int(rebalance_days))
    held = pd.Series(0.0, index=prices.columns)
    since = step  # decide on the first eligible bar
    for i, ts in enumerate(prices.index):
        # The book decided on PRIOR bars is what is held into this one.
        weights.iloc[i] = held.values
        if start is not None and ts < start:
            continue
        since += 1
        if since < step:
            continue
        ranked = mom.loc[ts][ok.loc[ts]].dropna()
        if len(ranked) < (k if long_only else 2 * k):
            continue
        since = 0
        order = ranked.sort_values(ascending=(signal == "reversal"))
        nxt = pd.Series(0.0, index=prices.columns)
        for sym in order.index[:k]:
            nxt[sym] = 1.0 / k
        if not long_only:
            for sym in order.index[-k:]:
                nxt[sym] = -1.0 / k
        held = nxt
    return weights


def backtest(prices, weights, fee):
    """Equity curve of the weighted book, charging `fee` on TURNOVER (both entries and exits). Starts at
    1.0. A symbol with no bar contributes no return, so a hole is cash rather than a fabricated move."""
    if prices.empty:
        return pd.Series(dtype=float)
    rets = step_returns(prices)
    equity = [1.0]
    prev_w = pd.Series(0.0, index=prices.columns)
    for i in range(1, len(prices.index)):
        w = weights.iloc[i]
        gross = float((w * rets.iloc[i]).sum())
        turnover = float((w - prev_w).abs().sum())
        equity.append(equity[-1] * (1.0 + gross) * (1.0 - fee * turnover))
        prev_w = w
    return pd.Series(equity, index=prices.index)


def basket_curve(prices, tradeable, fee):
    """The benchmark: buy-and-hold the EQUAL-WEIGHT tradeable universe (rebalanced as symbols become
    tradeable), charged one entry's worth of fee. `return_vs_hold_pct` is measured against this, not
    against any single asset."""
    if prices.empty:
        return pd.Series(dtype=float)
    rets = step_returns(prices)
    equity = [1.0 * (1.0 - fee)]
    for i in range(1, len(prices.index)):
        live = tradeable.iloc[i]
        n = int(live.sum())
        step = float(rets.iloc[i][live].sum() / n) if n else 0.0
        equity.append(equity[-1] * (1.0 + step))
    return pd.Series(equity, index=prices.index)


def _load_universe(symbols, pairs):
    """Daily bars per symbol over the requested (year, month) pairs. A symbol with nothing on disk is
    dropped from the universe rather than silently ranked as flat."""
    frames = {}
    for symbol in symbols:
        inst = data_catalog.instrument(symbol)
        directory = inst.directory if inst else "binance"
        paths = [f"{directory}/{symbol}-1d-{y}-{m}.json" for (y, m) in pairs]
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            continue
        df = pd.concat([pd.read_json(p) for p in paths], ignore_index=True)
        if "timestamp_close" not in df.columns or "price" not in df.columns:
            continue
        frames[symbol] = df[["timestamp_close", "price"]]
    return frames


def run(cfg):
    """Run one screen cell and return a trainer-contract RunSummary."""
    universe_id = str(cfg.get("universe", DEFAULT_UNIVERSE))
    symbols = UNIVERSES.get(universe_id)
    if not symbols:
        raise SystemExit(f"unknown universe {universe_id!r}; choose one of {sorted(UNIVERSES)}")
    lookback = int(cfg.get("lookback", 90))
    k = int(cfg.get("k", 3))
    rebalance_days = int(cfg.get("rebalance_days", 21))
    long_only = bool(cfg.get("long_only", False))
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        # build_weights raises on this too, but only after the universe has been read off disk and only as a
        # ValueError traceback. A cell is launched from a swept config file, so a bad lever must fail the
        # same way a bad `universe` does — one clean line, before any work — rather than as a stack trace
        # buried in a run log that someone then has to attribute to the right lever.
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    if lookback < 1:
        raise SystemExit(f"lookback must be >= 1; got {lookback} (a non-positive window reads the future)")
    fee = float(cfg.get("transaction_fee", 0.0002))

    train_pairs, test_pairs, window = resolve_walk_forward_window(cfg)
    frames = _load_universe(symbols, list(train_pairs) + list(test_pairs))
    if len(frames) < 2:
        raise SystemExit(f"cross-sectional needs >=2 symbols on disk; found {len(frames)}")
    prices = align_prices(frames)
    # The signal may look back into the TRAIN span, but only the TEST window is accounted — the same
    # out-of-sample rule the single-asset line reports under.
    test_start = pd.to_datetime(f"{test_pairs[0][0]}-{test_pairs[0][1]:02d}-01")
    weights = build_weights(prices, lookback, k, rebalance_days, long_only, signal, start=test_start)
    oos = prices.index >= test_start
    prices_oos = prices[oos]
    equity = backtest(prices_oos, weights[oos], fee)
    basket = basket_curve(prices_oos, tradeable_mask(prices, lookback)[oos], fee)

    total_return = float(equity.iloc[-1] - 1.0) * 100 if len(equity) else 0.0
    hold_return = float(basket.iloc[-1] - 1.0) * 100 if len(basket) else 0.0
    turnover = float(weights[oos].diff().abs().sum().sum())
    n_rebalances = int((weights[oos].diff().abs().sum(axis=1) > 1e-12).sum())
    metrics = {
        "total_return_pct": _finite(total_return),
        "hold_return_pct": _finite(hold_return),
        "return_vs_hold_pct": _finite(total_return - hold_return),
        "n_trades": n_rebalances,
        "turnover": _finite(turnover),
        "realized_cost_bps": _finite(turnover * fee * 10000),
        "universe_size": int(prices.shape[1]),
        "bars": int(len(prices_oos)),
    }
    metrics.update(_oos_stats(list(equity.values)))
    metrics.update(_max_drawdown_pct(list(equity.values)))
    metrics.update(_capture_stats(list(equity.values), list(basket.values)))
    summary = {
        "objective": _finite(total_return),
        "metrics": metrics,
        "dataset": {
            "asset": f"{universe_id}({prices.shape[1]})",
            "timeframe": "1d",
            "candles": int(len(prices_oos)),
            "from": str(prices_oos.index[0]) if len(prices_oos) else "",
            "to": str(prices_oos.index[-1]) if len(prices_oos) else "",
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
    parser = argparse.ArgumentParser(description="Cross-sectional long/short screen (B1 §1)")
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    with open(args.config_json) as fh:
        cfg = json.load(fh)
    summary = run(cfg)
    with open(args.summary_out, "w") as fh:
        json.dump(summary, fh)
    print(f"objective(total_return_pct)={summary['objective']:.4f} status=ok -> {args.summary_out}")


if __name__ == "__main__":
    main()
