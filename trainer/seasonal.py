"""Calendar / seasonal anomalies (the published-anomaly battery).

The famous free calendar effects: the TURN-OF-THE-MONTH (Ariel 1987; Lakonishok-Smidt 1988 — equity returns
concentrate in the last trading day of a month plus the first few of the next), SELL-IN-MAY / Halloween
(Bouman-Jacobsen 2002 — Nov-Apr beats May-Oct), and the MONDAY effect (French 1980 — Monday returns are
negative). Each is a deterministic date rule; this trades the calendar SPREAD — LONG the claimed-good window,
SHORT its complement (`seasonal`), or the exact negation (`seasonal_inverse`, the mirror control) — so the
DSR-deflated oos_sharpe of the spread directly measures the seasonal edge net of cost, isolating it from the
market's drift. Default universe is SPY (the equity index where these anomalies are defined); the book spreads
equally across a multi-asset universe when one is chosen.

The leakage surface is unusual and pinned in test_seasonal.py: a calendar position is a PURE FUNCTION OF THE
DATES and must be completely price-independent (corrupting every price leaves the book byte-identical) — a
rule that ever reads price is a bug or a disguised different strategy. Turnover (a few switches per year) is
costed. PRE-REGISTERED EXPECTATION: refutation net of cost OOS — these effects are decades-old, publicised,
and largely decayed/arbitraged (McLean-Pontiff 2016), and the long/short spread must overcome the switching
cost; a survivor across a majority of the deep-history windows would be a first cost-surviving edge.
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
    step_returns,
    tradeable_mask,
    _load_universe,
)

UNIVERSES = {
    "spy": ["SPY"],
    "equity_rates": ["SPY", "TLT", "IEF"],
    "diversified": ["GOLD", "SILVER", "COPPER", "WTI", "NATGAS", "CORN", "WHEAT", "SPY", "TLT", "IEF", "UUP"],
}
DEFAULT_UNIVERSE = "spy"

# The published rule (LONG the claimed-good window, SHORT its complement) and its exact negation (the mirror
# control, pre-registered so the winning direction cannot be cherry-picked). An unknown value is refused.
SIGNALS = ("seasonal", "seasonal_inverse")
DEFAULT_SIGNAL = "seasonal"

RULES = ("turn_of_month", "sell_in_may", "day_of_week")
DEFAULT_RULE = "turn_of_month"

# Sell-in-May "good" months (November through April).
_WINTER_MONTHS = frozenset({11, 12, 1, 2, 3, 4})


def calendar_position(index, rule, tom_days=3):
    """+1 inside the rule's claimed-good window, -1 in its complement, indexed by `index`. A pure function of
    the calendar — it never touches a price, so it is causal by construction (the date of a future bar is
    known in advance)."""
    idx = pd.DatetimeIndex(index)
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; choose one of {sorted(RULES)}")
    if rule == "sell_in_may":
        good = idx.month.isin(_WINTER_MONTHS)
    elif rule == "day_of_week":
        good = idx.weekday != 0  # Monday == 0 is the shorted day
    else:  # turn_of_month
        s = pd.Series(np.arange(len(idx)), index=idx)
        ym = pd.Index([(d.year, d.month) for d in idx])
        rank_in_month = s.groupby(ym).cumcount()  # 0-based position within the month
        is_first_n = rank_in_month < int(tom_days)
        # last trading day of the month = the max date within each (year, month) group
        last_of_month = s.groupby(ym).transform("max") == s.values
        good = (is_first_n | last_of_month).values
    return pd.Series(np.where(good, 1.0, -1.0), index=idx)


def build_weights(prices, rule=DEFAULT_RULE, signal=DEFAULT_SIGNAL, tom_days=3, start=None):
    """The book HELD INTO each bar, spread equally across the universe. Each leg is EXPOSURE-BALANCED by its
    calendar frequency — long the good window at +0.5/frac_in, short the complement at -0.5/frac_out — so the
    time-averaged net exposure is ZERO. This isolates the seasonal DIFFERENTIAL (is the per-day return higher
    in the window?) from the market's drift, which an unequal-window long/short would otherwise confound (the
    turn-of-month window is only ~4 of ~21 days, so a naive spread would be net-short the market and simply
    lose in bull years). The balancing frequency is a calendar constant (date-derived, never price), so the
    book stays a pure function of the calendar. `signal` "seasonal" is the published rule, "seasonal_inverse"
    its exact negation. Unknown levers are errors, not fallbacks."""
    if signal not in SIGNALS:
        raise ValueError(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    pos = calendar_position(prices.index, rule, tom_days)  # also validates rule
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if prices.empty:
        return weights
    good = pos.values > 0
    frac_in = float(good.mean())
    frac_out = 1.0 - frac_in
    if frac_in <= 0.0 or frac_out <= 0.0:
        bar_w = np.zeros(len(pos))  # a degenerate window (all or none) has no tradeable spread
    else:
        bar_w = np.where(good, 0.5 / frac_in, -0.5 / frac_out)
    if signal == "seasonal_inverse":
        bar_w = -bar_w
    n = max(1, prices.shape[1])
    for sym in prices.columns:
        weights[sym] = bar_w / n
    if start is not None:
        weights.loc[prices.index < start] = 0.0
    return weights


def run(cfg):
    """Run one seasonal cell and return a trainer-contract RunSummary."""
    universe_id = str(cfg.get("universe", DEFAULT_UNIVERSE))
    symbols = UNIVERSES.get(universe_id)
    if not symbols:
        raise SystemExit(f"unknown universe {universe_id!r}; choose one of {sorted(UNIVERSES)}")
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    rule = str(cfg.get("rule", DEFAULT_RULE))
    if rule not in RULES:
        raise SystemExit(f"unknown rule {rule!r}; choose one of {sorted(RULES)}")
    tom_days = int(cfg.get("tom_days", 3))
    fee = float(cfg.get("transaction_fee", 0.0005))

    train_pairs, test_pairs, window = resolve_walk_forward_window(cfg)
    frames = _load_universe(symbols, list(train_pairs) + list(test_pairs))
    if not frames:
        raise SystemExit(f"seasonal needs >=1 symbol on disk; found {len(frames)}")
    prices = align_prices(frames)
    test_start = pd.to_datetime(f"{test_pairs[0][0]}-{test_pairs[0][1]:02d}-01")
    weights = build_weights(prices, rule, signal, tom_days, start=test_start)
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
    parser = argparse.ArgumentParser(description="Calendar / seasonal anomalies (published-anomaly battery)")
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
