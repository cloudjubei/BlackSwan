"""Baltussen, Swinkels & van Vliet (2021), "Global Factor Premiums" — the paper's FOIL (published-anomaly battery).

Baltussen et al. is the one prior HUMAN multi-asset unified test: they report six style premiums (trend,
momentum, value, carry, seasonality, low-beta) as robust across equity/bond/commodity/currency over centuries.
Our contribution is the direct, explained CONTRADICTION — the same style factors, subjected to our discipline
(realistic per-trade cost, Deflated-Sharpe / best-of-N multiplicity, mutation-proven point-in-time guards, and
deep-history OOS), collapse. This module reproduces the factors feasible on the free survivorship-free panel
and — the headline foil — Baltussen's DIVERSIFIED equal-risk combination whose robustness is his central claim.

Deliberately thin: it REUSES the already-mutation-proven build_weights of tsmom (trend), xsection (momentum,
and VALUE = long-run cross-sectional reversal at a 5-year horizon, the commodity-value signal) and lowvol
(low-beta), so their leakage guards carry over verbatim; the only new object is the unit-gross DIVERSIFIED
average, whose causality (a linear combination of causal books) is pinned in test_globalfactors.py. SCOPE:
Baltussen's SEASONALITY is tested separately (the `seasonal` probe — disproved) and CARRY is data-gated on
free data (commodity carry ~ the roll yield, tested by the `roll` probe — disproved; equity/FX/bond carry need
term-structure / rates / FX not in the free panel), so this covers four of the six style factors directly plus
the diversified combination. PRE-REGISTERED EXPECTATION: the individual factors and their diversified
combination fail the DSR gate net of cost OOS, contradicting Baltussen once frictions + post-2008 history are
imposed uniformly.
"""

import argparse
import json

import pandas as pd

from trainer import lowvol, tsmom, xsection
from trainer.summary import (
    _capture_stats,
    _finite,
    _max_drawdown_pct,
    _oos_stats,
    _provenance_fingerprint,
)
from trainer.walk_forward import resolve_walk_forward_window
from trainer.xsection import align_prices, backtest, basket_curve, tradeable_mask, _load_universe

UNIVERSES = {
    "diversified": ["GOLD", "SILVER", "COPPER", "WTI", "NATGAS", "CORN", "WHEAT", "SPY", "TLT", "IEF", "UUP"],
    "commodities": ["GOLD", "SILVER", "COPPER", "WTI", "NATGAS", "CORN", "WHEAT"],
    "financials": ["SPY", "TLT", "IEF", "UUP"],
}
DEFAULT_UNIVERSE = "diversified"

# The four Baltussen style factors feasible on free price data, plus the diversified equal-risk combination
# (his central robustness claim). An unknown factor is refused, never a fallback.
DIVERSIFIED_LEGS = ("trend", "momentum", "value", "lowbeta")
FACTORS = ("trend", "momentum", "value", "lowbeta", "diversified")
DEFAULT_FACTOR = "diversified"

# "published" is the Baltussen direction; "inverse" is the exact negation (mirror control, pre-registered so
# the winning direction cannot be cherry-picked).
SIGNALS = ("published", "inverse")
DEFAULT_SIGNAL = "published"


def _unit_gross(w):
    """Scale each bar's book to unit gross (sum of |weights| = 1); a flat/warm-up bar (gross 0) stays flat.
    A per-row operation — it never mixes bars, so it preserves the causality of whatever it scales."""
    g = w.abs().sum(axis=1)
    return w.div(g.where(g > 0, 1.0), axis=0)


def factor_weights(prices, factor, start=None, lookback=252, value_lookback=1260, span=252, k=3, rebalance_days=21, vol_span=63):
    """The long/short book for one Baltussen style factor (or the diversified combination), decided on
    past-only signals and applied next bar (inherited from the reused build_weights). trend = time-series sign
    of the trailing return (tsmom); momentum = cross-sectional 12m winners-minus-losers (xsection); value =
    cross-sectional LONG-RUN (5y) reversal (xsection reversal at `value_lookback`); lowbeta = long low-beta /
    short high-beta (lowvol). An unknown factor is an error."""
    if factor == "trend":
        return tsmom.build_weights(prices, lookback, rebalance_days, "trend", "invvol", vol_span, False, start=start)
    if factor == "momentum":
        return xsection.build_weights(prices, lookback, k, rebalance_days, False, "momentum", start=start)
    if factor == "value":
        return xsection.build_weights(prices, value_lookback, k, rebalance_days, False, "reversal", start=start)
    if factor == "lowbeta":
        return lowvol.build_weights(prices, span, rebalance_days, "lowrisk", "beta", k, start=start)
    if factor == "diversified":
        legs = [_unit_gross(factor_weights(prices, f, start, lookback, value_lookback, span, k, rebalance_days, vol_span)) for f in DIVERSIFIED_LEGS]
        combined = sum(legs) / len(legs)
        return _unit_gross(combined)
    raise ValueError(f"unknown factor {factor!r}; choose one of {sorted(FACTORS)}")


def build_weights(prices, factor=DEFAULT_FACTOR, signal=DEFAULT_SIGNAL, start=None, **kw):
    """The factor book, negated under signal='inverse' (the mirror control)."""
    if signal not in SIGNALS:
        raise ValueError(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    w = factor_weights(prices, factor, start=start, **kw)
    return -w if signal == "inverse" else w


def run(cfg):
    """Run one global-factor cell and return a trainer-contract RunSummary."""
    universe_id = str(cfg.get("universe", DEFAULT_UNIVERSE))
    symbols = UNIVERSES.get(universe_id)
    if not symbols:
        raise SystemExit(f"unknown universe {universe_id!r}; choose one of {sorted(UNIVERSES)}")
    factor = str(cfg.get("factor", DEFAULT_FACTOR))
    if factor not in FACTORS:
        raise SystemExit(f"unknown factor {factor!r}; choose one of {sorted(FACTORS)}")
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    kw = dict(
        lookback=int(cfg.get("lookback", 252)),
        value_lookback=int(cfg.get("value_lookback", 1260)),
        span=int(cfg.get("span", 252)),
        k=int(cfg.get("k", 3)),
        rebalance_days=int(cfg.get("rebalance_days", 21)),
        vol_span=int(cfg.get("vol_span", 63)),
    )
    fee = float(cfg.get("transaction_fee", 0.0005))

    train_pairs, test_pairs, window = resolve_walk_forward_window(cfg)
    frames = _load_universe(symbols, list(train_pairs) + list(test_pairs))
    if len(frames) < 2:
        raise SystemExit(f"global factors need >=2 symbols on disk; found {len(frames)}")
    prices = align_prices(frames)
    test_start = pd.to_datetime(f"{test_pairs[0][0]}-{test_pairs[0][1]:02d}-01")
    weights = build_weights(prices, factor, signal, start=test_start, **kw)
    oos = prices.index >= test_start
    prices_oos = prices[oos]
    equity = backtest(prices_oos, weights[oos], fee)
    basket = basket_curve(prices_oos, tradeable_mask(prices, 1)[oos], fee)

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
    parser = argparse.ArgumentParser(description="Global factor premiums (Baltussen 2021) — published-anomaly battery")
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
