"""Time-series momentum / trend-following (the published-anomaly battery flagship).

The single most-defended free anomaly: Moskowitz, Ooi & Pedersen (2012) show that the SIGN of an asset's own
trailing 12-month return predicts its next-month return across ~58 futures, and the entire managed-futures /
CTA industry is built on it. Unlike the cross-sectional line ("will A beat B?") this asks the time-series
question ("is THIS market trending up or down?") and takes an absolute long/short book — so its exposure is
directional, not market-neutral. Deliberately crude and deterministic (the published rule, no training, no
RL): each asset is long when its trailing `lookback` return is positive, short when negative, sized either
equal-notional or inverse-vol (risk parity), rebalanced every `rebalance_days` and held between. The whole
portfolio is one equity curve emitting the SAME metric vocabulary as every other line (summary.py verbatim),
so the DSR gate, lenses and scorecard read it unchanged.

The correctness surface is the cross-sectional line's (a misaligned N-symbol join fabricates P&L; a symbol
ranked before it lists is survivorship bias — both reused verbatim from xsection) PLUS two the sign-of-trend
rule adds and test_tsmom.py pins: the vol scaler must size on PAST-only volatility, and the trend sign must
be read a bar before it is traded. The pre-registered expectation is refutation net of cost OOS (trend
following endured a long drawdown post-2011 and public rules decay — McLean-Pontiff 2016), but the arm is
free to survive: a positive DSR-deflated Sharpe in a majority of the deep-history windows would be the
program's first cost-surviving edge, and the mirror (`trend_inverse`) is pre-registered to block sign
cherry-picking.
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
from trainer.walk_forward import resolve_walk_forward_window
from trainer.xsection import (
    align_prices,
    backtest,
    basket_curve,
    momentum,
    step_returns,
    tradeable_mask,
    _load_universe,
)

# Pre-registered, never tuned on results (choosing symbols after seeing them is selection-on-test). Each is a
# survivorship-free persistent market/ETF with daily bars from 2006: commodities + equity + rates + a dollar
# proxy — the diversified futures-like basket Moskowitz's test rests on. US-session instruments only; 24/7
# crypto onto a session calendar is the hardest alignment case and is left to its own basket.
UNIVERSES = {
    "diversified": ["GOLD", "SILVER", "COPPER", "WTI", "NATGAS", "CORN", "WHEAT", "SPY", "TLT", "IEF", "UUP"],
    "commodities": ["GOLD", "SILVER", "COPPER", "WTI", "NATGAS", "CORN", "WHEAT"],
    "financials": ["SPY", "TLT", "IEF", "UUP"],
}
DEFAULT_UNIVERSE = "diversified"

# "trend" is the published rule (long positive trailing return, short negative); "trend_inverse" is its exact
# sign negation, pre-registered as the mirror control so the winning direction cannot be cherry-picked after
# the fact. An unrecognised value is an error, never a fallback — a typo that quietly ran trend would file a
# trend result under an inverse label and corrupt the evidence trail.
SIGNALS = ("trend", "trend_inverse")
DEFAULT_SIGNAL = "trend"

# "equal" sizes every held leg to the same notional; "invvol" sizes inverse to each leg's PAST-only trailing
# volatility (risk parity, Moskowitz's own sizing). Both normalise to unit gross so the equity curve is
# leverage-free and Sharpe (scale-invariant) is unaffected by the choice of gross.
WEIGHT_SCHEMES = ("equal", "invvol")
DEFAULT_WEIGHT_SCHEME = "invvol"


def trend_sign(prices, lookback):
    """+1 where the trailing `lookback`-bar return is positive, -1 where negative, 0 where flat/undefined.
    Backward-looking only (the value at t uses bars at or before t) — it reuses the cross-sectional line's
    trailing-return, so the same causality property carries over."""
    mom = momentum(prices, lookback)
    return np.sign(mom)


def trailing_vol(prices, span):
    """Per-asset trailing return volatility over `span` bars, measured on each symbol's OWN step returns.
    The value at t uses returns at or before t (a rolling window, no future bars), so sizing on it a bar
    later never peeks. NaN until a symbol has enough of its own history to estimate."""
    if prices.empty:
        return prices
    rets = step_returns(prices)
    return rets.rolling(int(span), min_periods=max(2, int(span) // 2)).std()


def build_weights(
    prices,
    lookback,
    rebalance_days,
    signal=DEFAULT_SIGNAL,
    weight_scheme=DEFAULT_WEIGHT_SCHEME,
    vol_span=63,
    long_only=False,
    start=None,
):
    """The book HELD INTO each bar. A trend sign computed from data up to t is applied to the NEXT step's
    return (row t carries what was decided at t-1), so a signal is never traded on its own bar. Rebalance
    every `rebalance_days` bars; between rebalances the book is held. Weights normalise to unit gross so the
    curve is leverage-free.

    `signal` chooses the direction: "trend" long the up-trenders / short the down-trenders, "trend_inverse"
    the exact negation. `weight_scheme` chooses sizing: "equal" same notional per leg, "invvol" inverse to
    each leg's past-only vol. Unknown levers and a non-positive lookback are errors, not fallbacks — the
    former corrupts the evidence trail, the latter ratios against a bar that has not happened and reads the
    future while every downstream guard waves it through."""
    if signal not in SIGNALS:
        raise ValueError(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    if weight_scheme not in WEIGHT_SCHEMES:
        raise ValueError(f"unknown weight_scheme {weight_scheme!r}; choose one of {sorted(WEIGHT_SCHEMES)}")
    if int(lookback) < 1:
        raise ValueError(f"lookback must be >= 1; got {lookback!r} (a non-positive window reads the future)")
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if prices.empty:
        return weights
    sign = trend_sign(prices, lookback)
    if signal == "trend_inverse":
        sign = -sign
    ok = tradeable_mask(prices, lookback)
    vol = trailing_vol(prices, vol_span) if weight_scheme == "invvol" else None
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
        s = sign.loc[ts][ok.loc[ts]].dropna()
        s = s[s != 0.0]
        if long_only:
            s = s[s > 0.0]
        if s.empty:
            continue
        since = 0
        if weight_scheme == "invvol":
            v = vol.loc[ts].reindex(s.index)
            inv = 1.0 / v.where(v > 0)
            inv = inv.dropna()
            s = s.reindex(inv.index)
            if s.empty or inv.sum() <= 0:
                continue
            mag = inv / inv.sum()
        else:
            mag = pd.Series(1.0 / len(s), index=s.index)
        nxt = pd.Series(0.0, index=prices.columns)
        nxt[s.index] = s.values * mag.values
        held = nxt
    return weights


def run(cfg):
    """Run one trend cell and return a trainer-contract RunSummary."""
    universe_id = str(cfg.get("universe", DEFAULT_UNIVERSE))
    symbols = UNIVERSES.get(universe_id)
    if not symbols:
        raise SystemExit(f"unknown universe {universe_id!r}; choose one of {sorted(UNIVERSES)}")
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    weight_scheme = str(cfg.get("weight_scheme", DEFAULT_WEIGHT_SCHEME))
    if weight_scheme not in WEIGHT_SCHEMES:
        raise SystemExit(f"unknown weight_scheme {weight_scheme!r}; choose one of {sorted(WEIGHT_SCHEMES)}")
    lookback = int(cfg.get("lookback", 252))
    if lookback < 1:
        raise SystemExit(f"lookback must be >= 1; got {lookback} (a non-positive window reads the future)")
    rebalance_days = int(cfg.get("rebalance_days", 21))
    vol_span = int(cfg.get("vol_span", 63))
    long_only = bool(cfg.get("long_only", False))
    fee = float(cfg.get("transaction_fee", 0.0005))

    train_pairs, test_pairs, window = resolve_walk_forward_window(cfg)
    frames = _load_universe(symbols, list(train_pairs) + list(test_pairs))
    if len(frames) < 2:
        raise SystemExit(f"time-series momentum needs >=2 symbols on disk; found {len(frames)}")
    prices = align_prices(frames)
    test_start = pd.to_datetime(f"{test_pairs[0][0]}-{test_pairs[0][1]:02d}-01")
    weights = build_weights(
        prices, lookback, rebalance_days, signal, weight_scheme, vol_span, long_only, start=test_start
    )
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
    except (Exception, SystemExit):
        pass
    return summary


def main():
    parser = argparse.ArgumentParser(description="Time-series momentum / trend-following (published-anomaly battery)")
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
