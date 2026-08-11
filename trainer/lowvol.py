"""Low-volatility / Betting-Against-Beta (the published-anomaly battery).

The low-risk anomaly (Baker-Haugen; Frazzini-Pedersen 2014, "Betting Against Beta") is one of the most-cited
"it survives" claims: low-risk assets deliver higher RISK-ADJUSTED returns than high-risk ones, so a book
LONG the low-risk names and SHORT the high-risk names earns a positive alpha. This ranks a survivorship-free
basket by a PAST-only risk score — trailing realised volatility (the low-vol anomaly) or trailing beta to the
equal-weight basket (BAB) — and holds a dollar-neutral long-low / short-high book (`lowrisk`), or its exact
negation (`lowrisk_inverse`, the mirror control), rebalanced every `rebalance_days` and turnover-costed. The
whole portfolio is one equity curve emitting summary.py's metric vocabulary, so the DSR gate reads it
unchanged; the decider is the DSR-deflated oos_sharpe across a majority of the deep-history windows.

The correctness surface is the cross-sectional line's (misaligned join fabricates P&L; a symbol ranked before
it lists is survivorship bias — reused verbatim) PLUS the two leakage surfaces the rule adds and test_lowvol.py
pins: the risk score must be computed on PAST-only bars and read a bar before it trades, for BOTH rank keys.
PRE-REGISTERED EXPECTATION: refutation net of cost OOS — on a small, mostly-directional commodity+rates basket
the low-risk/high-risk split collapses toward a rates-vs-commodities bet, and public low-vol strategies
crowded/decayed; a survivor across a majority of windows (thesis not inverse) would be a first cost-surviving
edge.
"""

import argparse
import json

import numpy as np
import pandas as pd

from trainer.summary import (
    _capture_stats,
    _finite,
    _max_drawdown_pct,
    _oos_stats,
    _provenance_fingerprint,
)
from trainer.tsmom import trailing_vol
from trainer.walk_forward import resolve_walk_forward_window
from trainer.xsection import (
    align_prices,
    backtest,
    basket_curve,
    step_returns,
    tradeable_mask,
    _load_universe,
)

UNIVERSES = {
    "diversified": ["GOLD", "SILVER", "COPPER", "WTI", "NATGAS", "CORN", "WHEAT", "SPY", "TLT", "IEF", "UUP"],
    "commodities": ["GOLD", "SILVER", "COPPER", "WTI", "NATGAS", "CORN", "WHEAT"],
    "financials": ["SPY", "TLT", "IEF", "UUP"],
}
DEFAULT_UNIVERSE = "diversified"

# "lowrisk" is the published rule (long the low-risk names, short the high-risk); "lowrisk_inverse" is its exact
# negation, pre-registered as the mirror control. An unrecognised value is an error, never a fallback.
SIGNALS = ("lowrisk", "lowrisk_inverse")
DEFAULT_SIGNAL = "lowrisk"

# "vol" ranks by trailing realised volatility (the low-volatility anomaly); "beta" ranks by trailing beta to
# the equal-weight basket (Betting-Against-Beta). Both past-only. An unrecognised key is refused.
RANK_KEYS = ("vol", "beta")
DEFAULT_RANK_KEY = "beta"


def trailing_beta(prices, span):
    """Per-asset trailing beta to the EQUAL-WEIGHT basket, over `span` bars, using only bars at or before t.
    beta_i(t) = cov(r_i, r_mkt) / var(r_mkt) on the trailing window, where r_mkt is the cross-sectional mean
    step return. NaN until a symbol has enough of its own history. Past-only, so sizing on it a bar later
    never peeks."""
    if prices.empty:
        return prices
    rets = step_returns(prices)
    mkt = rets.mean(axis=1)
    span = int(span)
    mp = max(2, span // 2)
    var = mkt.rolling(span, min_periods=mp).var()
    out = {}
    for symbol in prices.columns:
        cov = rets[symbol].rolling(span, min_periods=mp).cov(mkt)
        out[symbol] = cov / var.where(var > 0)
    return pd.DataFrame(out, index=prices.index)


def _risk_score(prices, rank_by, span):
    if rank_by not in RANK_KEYS:
        raise ValueError(f"unknown rank_by {rank_by!r}; choose one of {sorted(RANK_KEYS)}")
    return trailing_vol(prices, span) if rank_by == "vol" else trailing_beta(prices, span)


def build_weights(
    prices, span, rebalance_days, signal=DEFAULT_SIGNAL, rank_by=DEFAULT_RANK_KEY, k=3, start=None
):
    """The book HELD INTO each bar. A risk score computed from data up to t is applied to the NEXT step's
    return (row t carries what was decided at t-1), so the score is never traded on its own bar. `signal`
    "lowrisk" longs the k LOWEST-score / shorts the k HIGHEST-score names, "lowrisk_inverse" the exact
    negation; `rank_by` chooses the score. Dollar-neutral equal-weight (1/k per leg, gross 2, net 0).
    Unknown levers are errors, not fallbacks — they would file one rule's result under another's label."""
    if signal not in SIGNALS:
        raise ValueError(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    score = _risk_score(prices, rank_by, span)  # also validates rank_by
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if prices.empty:
        return weights
    ok = tradeable_mask(prices, span)
    k = max(1, int(k))
    step = max(1, int(rebalance_days))
    held = pd.Series(0.0, index=prices.columns)
    since = step
    for i, ts in enumerate(prices.index):
        weights.iloc[i] = held.values
        if start is not None and ts < start:
            continue
        since += 1
        if since < step:
            continue
        ranked = score.loc[ts][ok.loc[ts]].dropna()
        if len(ranked) < 2 * k:
            continue
        since = 0
        order = ranked.sort_values(ascending=True)  # lowest risk first
        low = list(order.index[:k])
        high = list(order.index[-k:])
        if signal == "lowrisk_inverse":
            low, high = high, low
        nxt = pd.Series(0.0, index=prices.columns)
        for sym in low:
            nxt[sym] = 1.0 / k
        for sym in high:
            nxt[sym] = -1.0 / k
        held = nxt
    return weights


def run(cfg):
    """Run one low-risk cell and return a trainer-contract RunSummary."""
    universe_id = str(cfg.get("universe", DEFAULT_UNIVERSE))
    symbols = UNIVERSES.get(universe_id)
    if not symbols:
        raise SystemExit(f"unknown universe {universe_id!r}; choose one of {sorted(UNIVERSES)}")
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    rank_by = str(cfg.get("rank_by", DEFAULT_RANK_KEY))
    if rank_by not in RANK_KEYS:
        raise SystemExit(f"unknown rank_by {rank_by!r}; choose one of {sorted(RANK_KEYS)}")
    span = int(cfg.get("span", 63))
    if span < 2:
        raise SystemExit(f"span must be >= 2; got {span} (a risk score needs at least two bars)")
    rebalance_days = int(cfg.get("rebalance_days", 21))
    k = int(cfg.get("k", 3))
    fee = float(cfg.get("transaction_fee", 0.0005))

    train_pairs, test_pairs, window = resolve_walk_forward_window(cfg)
    frames = _load_universe(symbols, list(train_pairs) + list(test_pairs))
    if len(frames) < 2 * k:
        raise SystemExit(f"low-risk needs >= 2*k symbols on disk; found {len(frames)} for k={k}")
    prices = align_prices(frames)
    test_start = pd.to_datetime(f"{test_pairs[0][0]}-{test_pairs[0][1]:02d}-01")
    weights = build_weights(prices, span, rebalance_days, signal, rank_by, k, start=test_start)
    oos = prices.index >= test_start
    prices_oos = prices[oos]
    equity = backtest(prices_oos, weights[oos], fee)
    basket = basket_curve(prices_oos, tradeable_mask(prices, span)[oos], fee)

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
    except (Exception, SystemExit):
        pass
    return summary


def main():
    parser = argparse.ArgumentParser(description="Low-volatility / Betting-Against-Beta (published-anomaly battery)")
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
