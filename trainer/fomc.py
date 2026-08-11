"""Pre-FOMC announcement drift (Lucca-Moench 2015) — the published-anomaly battery, event-calendar family.

Lucca & Moench (2015, JF) document that US equity returns are large and significant in the ~24 hours before
scheduled FOMC announcements; Kurov, Wolfe & Gilbert (2021) find the drift disappeared after publication. We
test it as an EVENT-CALENDAR SPREAD — exposure-balanced LONG the pre-FOMC trading day(s), SHORT the complement,
so the time-averaged net exposure is zero and the DSR-deflated oos_sharpe measures the pre-FOMC differential
isolated from market drift — on SPY (its equity home) and on BTC (the 24/7 crypto window, which has never been
cost+DSR-tested — the genuinely new angle). `drift` is the published direction (long pre-FOMC), `drift_inverse`
the mirror control.

The FOMC calendar is PUBLISHED a year ahead, so positioning on a pre-FOMC day is causal by construction; the
book is a PURE FUNCTION OF THE (scheduled) FOMC dates and never of price (mutation-proven in test_fomc.py).
Only SCHEDULED meetings are used (2020's March emergency cuts are excluded — an unscheduled cut has no
anticipatory pre-drift). PRE-REGISTERED EXPECTATION: refutation net of cost OOS — the equity drift decayed
post-2015 (Kurov 2021) and the crypto window is a fresh, most-likely-null test.
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
from trainer.xsection import align_prices, backtest, basket_curve, tradeable_mask, _load_universe

UNIVERSES = {"spy": ["SPY"], "btc": ["BTCUSDT"]}
DEFAULT_UNIVERSE = "spy"

SIGNALS = ("drift", "drift_inverse")
DEFAULT_SIGNAL = "drift"

# Scheduled FOMC announcement (statement) dates 2015-2024, from the Federal Reserve's FOMC calendars. Only
# regularly-scheduled meetings (2020's unscheduled March emergency cuts are excluded — no anticipatory drift).
FOMC_DATES = [
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
]


def fomc_position(index, fomc_dates, pre_days=1):
    """+1 on the `pre_days` trading days immediately BEFORE a scheduled FOMC announcement, -1 elsewhere; a pure
    function of the trading calendar and the (published) FOMC dates. For each announcement, the pre-window is
    the last `pre_days` trading days STRICTLY before it (works whether or not the announcement date is itself a
    trading day in the index)."""
    idx = pd.DatetimeIndex(index)
    good = np.zeros(len(idx), dtype=bool)
    ann = pd.DatetimeIndex(sorted(pd.to_datetime(d) for d in fomc_dates))
    for d in ann:
        before = np.flatnonzero(idx < d)  # positions strictly before the announcement
        if len(before):
            good[before[-int(pre_days):]] = True
    return pd.Series(np.where(good, 1.0, -1.0), index=idx)


def build_weights(prices, signal=DEFAULT_SIGNAL, pre_days=1, start=None):
    """The book HELD INTO each bar: the exposure-balanced pre-FOMC spread, spread equally across the universe.
    The long leg (pre-FOMC days) is scaled +0.5/frac_in and the short leg -0.5/frac_out, so the time-averaged
    net exposure is ZERO — isolating the pre-FOMC differential from market drift. Price-independent by
    construction. `drift` is the published direction, `drift_inverse` its exact negation. Unknown signals are
    errors, not fallbacks."""
    if signal not in SIGNALS:
        raise ValueError(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    pos = fomc_position(prices.index, FOMC_DATES, pre_days)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if prices.empty:
        return weights
    good = pos.values > 0
    frac_in = float(good.mean())
    frac_out = 1.0 - frac_in
    if frac_in <= 0.0 or frac_out <= 0.0:
        bar_w = np.zeros(len(pos))
    else:
        bar_w = np.where(good, 0.5 / frac_in, -0.5 / frac_out)
    if signal == "drift_inverse":
        bar_w = -bar_w
    n = max(1, prices.shape[1])
    for sym in prices.columns:
        weights[sym] = bar_w / n
    if start is not None:
        weights.loc[prices.index < start] = 0.0
    return weights


def run(cfg):
    """Run one pre-FOMC cell and return a trainer-contract RunSummary."""
    universe_id = str(cfg.get("universe", DEFAULT_UNIVERSE))
    symbols = UNIVERSES.get(universe_id)
    if not symbols:
        raise SystemExit(f"unknown universe {universe_id!r}; choose one of {sorted(UNIVERSES)}")
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    pre_days = int(cfg.get("pre_days", 1))
    fee = float(cfg.get("transaction_fee", 0.0005))

    train_pairs, test_pairs, window = resolve_walk_forward_window(cfg)
    frames = _load_universe(symbols, list(train_pairs) + list(test_pairs))
    if not frames:
        raise SystemExit(f"pre-FOMC needs >=1 symbol on disk; found {len(frames)}")
    prices = align_prices(frames)
    test_start = pd.to_datetime(f"{test_pairs[0][0]}-{test_pairs[0][1]:02d}-01")
    weights = build_weights(prices, signal, pre_days, start=test_start)
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
    parser = argparse.ArgumentParser(description="Pre-FOMC announcement drift (Lucca-Moench 2015) — published-anomaly battery")
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
