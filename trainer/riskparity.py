"""Cross-class risk-parity screen (Probe A) — the one portfolio curve this program never scored.

Every arm of this campaign so far has asked a directional question ("will THIS market go up?", "will A
outperform B?") or a sizing one ("how big should the book be?"). None has scored the plainest allocation
there is: hold a DIVERSIFIED, LONG-ONLY basket across asset classes — gold, Treasuries (long/intermediate),
the dollar, equities, crypto — vol-scaled so each leg contributes roughly equal risk, rebalanced every N
bars, net of realized cost. This screen scores exactly that, and asks whether it beats simply buy-and-holding
the equal-weight basket.

The expectation is PRE-REGISTERED and negative: a long-only diversified book is a long-BETA book. It cannot
be net-flat the way the cross-sectional fork is, so it will move with the risk-on/risk-off tape and is NOT
expected to beat the equal-weight basket on return. This screen MEASURES that rather than assuming it — the
honest read is the risk lens (oos_sharpe, sharpe_vs_hold, drawdown_vs_hold_pct), because risk parity earns
its keep by being STEADIER than naive weighting, not by out-returning it. return_vs_hold_pct is emitted as
the hypothesis benchmark (threshold 0) precisely so a clean NO is recorded, not so a YES is hoped for.

It is a NEW ENTRY POINT, deliberately crude, published, deterministic, no training, its own recordType so its
cells never mix with the RL run store. It REUSES the cross-sectional screen's primitives verbatim
(``align_prices``, ``step_returns``, ``tradeable_mask``, ``backtest``, ``basket_curve``, ``_load_universe``)
so an N-symbol join, a warm-up, and a fee-on-turnover mean exactly what they mean there, and it emits the
SAME summary.py equity-curve vocabulary so every existing gate, lens and scorecard reads it unchanged.

Correctness properties are pinned in test_riskparity.py before the screen is allowed to produce a number. An
inverse-vol book is trivially easy to write with one bar of lookahead — the moment a weight is sized from the
vol observed on the very bar its return is booked, the backtest knows tomorrow's risk regime and every number
it emits is fiction — so the strictly-past weighting is the single most important property in this module.
"""

import argparse
import json

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
    _load_universe,
    align_prices,
    backtest,
    basket_curve,
    tradeable_mask,
)

# The universe is PRE-REGISTERED and never tuned on results — choosing symbols after seeing them is
# selection-on-test. ``cross-class`` is the headline diversified book across five asset classes; ``macro``
# drops ETH (whose on-disk history begins 2022-01) so it can run on the deeper 2018+ windows without the
# universe silently shrinking on the early bars. GOLD lives in commodities/, SPY/TLT/IEF/UUP in etfs/, the
# crypto in binance/ — each symbol's directory resolves from data_catalog, exactly as the loader expects.
UNIVERSES = {
    "cross-class": ["GOLD", "TLT", "IEF", "UUP", "SPY", "BTCUSDT", "ETHUSDT"],
    "macro": ["GOLD", "TLT", "IEF", "UUP", "SPY", "BTCUSDT"],
}
DEFAULT_UNIVERSE = "cross-class"

# How the fully-invested long book is split over the tradeable set. ``inverse_vol`` is the risk-parity
# approximation the probe exists to score (w_i proportional to 1/v_i, so each leg contributes roughly equal
# risk); ``equal_weight`` is the reference arm (w_i = 1/n) and, at rebalance_days=1 with no fee, IS the basket
# benchmark. True equal-risk-contribution is deliberately NOT implemented — inverse-vol is the accepted proxy
# and an ERC solver would be over-engineering for a screen whose job is to be cheap and legible.
WEIGHTINGS = ("inverse_vol", "equal_weight")
DEFAULT_WEIGHTING = "inverse_vol"


def trailing_vol(prices, vol_window):
    """Realized vol per symbol at each bar: the stdev of that symbol's last ``vol_window`` per-step returns,
    over the symbol's OWN observations, using returns up to and INCLUDING the bar. Backward-looking only —
    the value at t uses bars at or before t.

    Computed on the symbol's dropna'd series so a calendar gap on one asset's clock (a market holiday, a
    weekend against 24/7 crypto) is SKIPPED rather than counted as a flat 0 return — a fabricated zero would
    understate the vol and, inverted, inflate that leg's weight. NaN until ``vol_window`` returns exist: the
    warm-up is a hole, never a zero, because a zero vol reads as "no risk" and demands maximum size. The
    stdev is left un-annualized (population, ddof=0); the sizing is a ratio of vols measured the same way, so
    any constant factor cancels in the normalization."""
    if prices.empty:
        return prices
    out = {}
    for symbol in prices.columns:
        s = prices[symbol].dropna()
        r = s / s.shift(1) - 1.0
        out[symbol] = r.rolling(int(vol_window)).std(ddof=0).reindex(prices.index)
    return pd.DataFrame(out, index=prices.index)


def _target_weights(live, vol_row, weighting, columns):
    """The long-only, fully-invested book for ONE rebalance bar over the tradeable set, or ``None`` when
    nothing is tradeable (the caller then HOLDS its previous book rather than writing zeros mid-stream).

    ``equal_weight`` splits 1/n over the tradeable names and never touches vol. ``inverse_vol`` sizes each leg
    proportional to 1/v_i and normalizes to sum 1 — the risk-parity approximation. A tradeable name whose vol
    is zero or non-finite is EXCLUDED from the inverse-vol sizing (an infinite reciprocal would claim the
    whole book), falling back to an equal split only in the pathological case where every leg is degenerate;
    with real daily data a positive vol is always present, so the fallback never fires in a real cell."""
    n = int(live.sum())
    if n == 0:
        return None
    if weighting == "equal_weight":
        w = pd.Series(0.0, index=columns)
        w[live] = 1.0 / n
        return w
    inv = pd.Series(0.0, index=columns)
    usable = live & vol_row.notna() & (vol_row > 0.0)
    inv[usable] = 1.0 / vol_row[usable]
    total = float(inv.sum())
    if total <= 0.0:
        w = pd.Series(0.0, index=columns)
        w[live] = 1.0 / n
        return w
    return inv / total


def build_weights(prices, vol_window, rebalance_days, weighting=DEFAULT_WEIGHTING, start=None):
    """The book HELD INTO each bar. A weight sized from vols observed through t is applied to the NEXT step's
    return (row t carries what was decided at t-1), so a leg is never sized on the very bar whose return it
    earns — the whole causality contract of the screen. Rebalance every ``rebalance_days`` bars; between
    rebalances the book is held constant, which is the turnover — and therefore the cost — control. Warm-up
    or not-yet-tradeable names get weight 0 and the book renormalizes over whatever IS tradeable, so it stays
    fully invested without ever emitting a NaN. ``start`` gates when DECISIONS may begin (the test window)
    while the vol estimator keeps its full formation history.

    ``vol_window`` must be >= 2: a stdev needs at least two returns, and a window of 1 makes every realized
    vol 0, which inverts to an infinite weight and, on the degenerate fallback, silently turns inverse_vol
    INTO equal_weight — an equal-weight result filed under a risk-parity label, which corrupts the evidence
    trail rather than merely losing money. An unrecognised ``weighting`` is refused for the same reason: a
    typo that quietly ranked as the default would mislabel every cell it produced."""
    if weighting not in WEIGHTINGS:
        raise ValueError(f"unknown weighting {weighting!r}; choose one of {sorted(WEIGHTINGS)}")
    if int(vol_window) < 2:
        raise ValueError(
            f"vol_window must be >= 2; got {vol_window!r} (a stdev needs at least two returns, and a "
            "window of 1 zeroes every vol and demands infinite weight)"
        )
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if prices.empty:
        return weights
    vol = trailing_vol(prices, vol_window)
    ok = tradeable_mask(prices, vol_window)
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
        target = _target_weights(ok.loc[ts], vol.loc[ts], weighting, prices.columns)
        if target is None:  # nothing tradeable yet — hold flat rather than write zeros over a live book
            continue
        since = 0
        held = target
    return weights


def _benchmark_metrics(equity, basket):
    """The risk lens this screen is honestly read through, built by pushing the BASKET curve through the SAME
    ``_oos_stats`` / ``_max_drawdown_pct`` helpers as the strategy curve so the two Sharpes and the two
    drawdowns are never measured two different ways.

    ``sharpe_vs_hold`` and ``drawdown_vs_hold_pct`` are both signed so better-than-hold is POSITIVE — drawdowns
    are negative percents, so a shallower one minus a deeper one is positive. These are the reads that can say
    "steadier than naive weighting" even when a long-beta book returns less than the basket (which it is
    pre-registered to). Empty (skippable via metrics.update) when either curve is too short for the helpers."""
    strat, bench = _oos_stats(equity), _oos_stats(basket)
    strat_dd, bench_dd = _max_drawdown_pct(equity), _max_drawdown_pct(basket)
    if not strat or not bench or not strat_dd or not bench_dd:
        return {}
    return {
        "hold_sharpe": bench["oos_sharpe"],
        "sharpe_vs_hold": _finite(strat["oos_sharpe"] - bench["oos_sharpe"]),
        "hold_max_drawdown_pct": bench_dd["max_drawdown_pct"],
        "drawdown_vs_hold_pct": _finite(strat_dd["max_drawdown_pct"] - bench_dd["max_drawdown_pct"]),
    }


def run(cfg):
    """Run one screen cell and return a trainer-contract RunSummary."""
    universe_id = str(cfg.get("universe", DEFAULT_UNIVERSE))
    symbols = UNIVERSES.get(universe_id)
    if not symbols:
        raise SystemExit(f"unknown universe {universe_id!r}; choose one of {sorted(UNIVERSES)}")
    vol_window = int(cfg.get("vol_window", 20))
    rebalance_days = int(cfg.get("rebalance_days", 21))
    weighting = str(cfg.get("weighting", DEFAULT_WEIGHTING))
    if weighting not in WEIGHTINGS:
        # build_weights raises on this too, but only after the universe has been read off disk. A cell is
        # launched from a swept config, so a bad lever must fail one clean line before any work — the same
        # way a bad universe does — rather than as a ValueError buried in a run log.
        raise SystemExit(f"unknown weighting {weighting!r}; choose one of {sorted(WEIGHTINGS)}")
    if vol_window < 2:
        raise SystemExit(f"vol_window must be >= 2; got {vol_window} (a stdev needs at least two returns)")
    fee = float(cfg.get("transaction_fee", 0.0002))

    train_pairs, test_pairs, window = resolve_walk_forward_window(cfg)
    frames = _load_universe(symbols, list(train_pairs) + list(test_pairs))
    if len(frames) < 2:
        raise SystemExit(f"risk-parity needs >=2 symbols on disk; found {len(frames)}")
    prices = align_prices(frames)
    # The vol estimator may look back into the TRAIN span for its formation window, but only the TEST
    # window's P&L is accounted — the same out-of-sample rule the rest of the line reports under.
    test_start = pd.to_datetime(f"{test_pairs[0][0]}-{test_pairs[0][1]:02d}-01")
    weights = build_weights(prices, vol_window, rebalance_days, weighting, start=test_start)
    oos = prices.index >= test_start
    prices_oos = prices[oos]
    weights_oos = weights[oos]
    equity = backtest(prices_oos, weights_oos, fee)
    basket = basket_curve(prices_oos, tradeable_mask(prices, vol_window)[oos], fee)

    total_return = float(equity.iloc[-1] - 1.0) * 100 if len(equity) else 0.0
    hold_return = float(basket.iloc[-1] - 1.0) * 100 if len(basket) else 0.0
    turnover = float(weights_oos.diff().abs().sum().sum())
    n_rebalances = int((weights_oos.diff().abs().sum(axis=1) > 1e-12).sum())
    metrics = {
        "total_return_pct": _finite(total_return),
        "hold_return_pct": _finite(hold_return),
        "return_vs_hold_pct": _finite(total_return - hold_return),
        "n_trades": n_rebalances,
        "turnover": _finite(turnover),
        "realized_cost_bps": _finite(turnover * fee * 10000),
        # A fully-invested book sits at exposure 1 on every traded bar; the mean over the test window reads
        # ~1 (pulled a hair below only by the flat opening bar). A value far from 1 means the book is
        # cash-heavy — its Sharpe would be measuring mostly cash — so this is a sanity check, not a lever.
        "mean_exposure": _finite(float(weights_oos.sum(axis=1).mean())) if len(weights_oos) else 0.0,
        "universe_size": int(prices.shape[1]),
        "bars": int(len(prices_oos)),
    }
    metrics.update(_oos_stats(list(equity.values)))
    metrics.update(_max_drawdown_pct(list(equity.values)))
    # beta / up_capture / down_capture are measured against the EQUAL-WEIGHT BASKET, not a single asset — the
    # honest "how much of the basket's beta does this book carry?" read for a long-only diversified portfolio.
    metrics.update(_capture_stats(list(equity.values), list(basket.values)))
    metrics.update(_benchmark_metrics(list(equity.values), list(basket.values)))
    summary = {
        # The objective is the RISK-adjusted read (oos_sharpe): a long-beta book is not judged on out-earning
        # the basket, so raw return is context (return_vs_hold_pct, the hypothesis benchmark) not the score.
        "objective": _finite(metrics.get("oos_sharpe", 0.0)),
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
    parser = argparse.ArgumentParser(description="Cross-class risk-parity screen (Probe A)")
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
