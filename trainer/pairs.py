"""Pairs / statistical arbitrage (the published-anomaly battery).

Distance pairs trading (Gatev-Goetzmann-Rouwenhorst 2006): over a FORMATION window normalise each asset to a
cumulative-return index, pick the k closest-moving pairs by sum-of-squared-distance, then in the TRADING
window fade divergences of each pair's spread — when the normalised spread stretches beyond `entry` formation
std-devs, SHORT the rich leg / LONG the cheap leg, and CLOSE on convergence (the spread crossing its formation
mean). It is a distinct MECHANISM from everything else in the battery — spread mean-reversion, not price
direction, and market-neutral by construction. `signal="meanrev"` is the published rule; `meanrev_inverse`
(divergence-chasing) is the exact negation, the mirror control. The portfolio is the equal-capital average of
the k pair books; it emits summary.py's metric vocabulary so the DSR gate reads it unchanged.

The leakage surface is pinned in test_pairs.py: pair SELECTION and the spread's mean/std come ONLY from the
formation window (choosing pairs on the trading window is peeking), the entry/exit state machine is exactly
the mean-reversion rule, and the book is one bar behind the spread it reads. PRE-REGISTERED EXPECTATION:
refutation net of cost OOS — GGR's edge decayed sharply after ~2002 as it was arbitraged, and a small daily
free basket has few genuinely-cointegrated pairs; a survivor across a majority of the deep-history windows
would be a first cost-surviving edge.
"""

import argparse
import itertools
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
    tradeable_mask,
    _load_universe,
)

UNIVERSES = {
    "diversified": ["GOLD", "SILVER", "COPPER", "WTI", "NATGAS", "CORN", "WHEAT", "SPY", "TLT", "IEF", "UUP"],
    "commodities": ["GOLD", "SILVER", "COPPER", "WTI", "NATGAS", "CORN", "WHEAT"],
    "financials": ["SPY", "TLT", "IEF", "UUP"],
}
DEFAULT_UNIVERSE = "diversified"

# The published rule (fade divergence) and its exact negation (chase divergence — the mirror control). An
# unknown value is refused, never a fallback.
SIGNALS = ("meanrev", "meanrev_inverse")
DEFAULT_SIGNAL = "meanrev"


def normalize(prices):
    """Each asset as a cumulative index anchored to its first observed price in the frame (a price relative,
    the GGR normalisation). Forward-holds within a symbol's own history for spread arithmetic; a symbol with
    no data stays NaN."""
    if prices.empty:
        return prices
    out = {}
    for sym in prices.columns:
        s = prices[sym]
        first = s.dropna()
        if first.empty:
            out[sym] = s
            continue
        out[sym] = s / first.iloc[0]
    return pd.DataFrame(out, index=prices.index)


def select_pairs(norm, k):
    """The k closest pairs by sum-of-squared-distance of their normalised series (GGR's distance rule). Only
    pairs with full overlapping data over the passed (formation) window are eligible. Deterministic tie-break
    by symbol name so the selection is reproducible."""
    cols = [c for c in norm.columns if norm[c].notna().all()]
    scored = []
    for a, b in itertools.combinations(sorted(cols), 2):
        ssd = float(((norm[a] - norm[b]) ** 2).sum())
        scored.append((ssd, a, b))
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    return [(a, b) for _ssd, a, b in scored[: max(1, int(k))]]


def pair_position_series(z, entry):
    """The mean-reversion state machine over a z-score path: FLAT until |z| exceeds `entry`, then SHORT the
    spread (-1) if z was high / LONG the spread (+1) if z was low, HELD until z crosses its mean (0), then
    flat again. Pure and deterministic — the trading rule with no data access."""
    entry = float(entry)
    pos = 0
    out = []
    for zt in z:
        if pos == 0:
            if zt > entry:
                pos = -1
            elif zt < -entry:
                pos = 1
        elif (pos == -1 and zt <= 0.0) or (pos == 1 and zt >= 0.0):
            pos = 0
        out.append(pos)
    return out


def build_weights(prices, formation_days, k, entry, signal=DEFAULT_SIGNAL, test_start=None):
    """The book HELD INTO each bar. Pairs are SELECTED and each spread's mean/std are estimated on the
    FORMATION window (the `formation_days` bars ending just before `test_start`) — strictly past — then each
    pair is traded over the test window by the mean-reversion state machine, the pair position decided at t-1
    and applied to bar t (a one-bar lag), converted to +/-0.5 on each leg (dollar-neutral) and averaged over
    the k pairs. `meanrev_inverse` negates the book. Unknown signals are errors, not fallbacks."""
    if signal not in SIGNALS:
        raise ValueError(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if prices.empty or test_start is None:
        return weights
    idx = prices.index
    form_mask = idx < test_start
    form = prices[form_mask]
    if int(formation_days) > 0:
        form = form.iloc[-int(formation_days):]
    if len(form) < 3:
        return weights
    norm_form = normalize(form)
    chosen = select_pairs(norm_form, k)
    if not chosen:
        return weights
    # Normalise the WHOLE series to the formation anchor so the trading spread is on the same scale as the
    # formation mean/std (both are relatives to each asset's first formation price).
    norm_all = normalize(prices[idx >= form.index[0]]).reindex(idx)
    kk = len(chosen)
    for a, b in chosen:
        spread_form = norm_form[a] - norm_form[b]
        mu = float(spread_form.mean())
        sd = float(spread_form.std())
        if not np.isfinite(sd) or sd <= 0:
            continue
        spread = norm_all[a] - norm_all[b]
        z = (spread - mu) / sd
        pos = pd.Series(pair_position_series(list(z.fillna(0.0).values), entry), index=idx)
        pos = pos.where(idx >= test_start, 0).shift(1).fillna(0.0)  # decide at t-1, hold into t
        if signal == "meanrev_inverse":
            pos = -pos
        weights[a] = weights[a] + pos * (0.5 / kk)
        weights[b] = weights[b] - pos * (0.5 / kk)
    return weights


def run(cfg):
    """Run one pairs cell and return a trainer-contract RunSummary."""
    universe_id = str(cfg.get("universe", DEFAULT_UNIVERSE))
    symbols = UNIVERSES.get(universe_id)
    if not symbols:
        raise SystemExit(f"unknown universe {universe_id!r}; choose one of {sorted(UNIVERSES)}")
    signal = str(cfg.get("signal", DEFAULT_SIGNAL))
    if signal not in SIGNALS:
        raise SystemExit(f"unknown signal {signal!r}; choose one of {sorted(SIGNALS)}")
    formation_days = int(cfg.get("formation_days", 252))
    k = int(cfg.get("k", 5))
    entry = float(cfg.get("entry", 2.0))
    if entry <= 0:
        raise SystemExit(f"entry must be > 0; got {entry} (a non-positive threshold trades every bar)")
    fee = float(cfg.get("transaction_fee", 0.0005))

    train_pairs, test_pairs, window = resolve_walk_forward_window(cfg)
    frames = _load_universe(symbols, list(train_pairs) + list(test_pairs))
    if len(frames) < 2:
        raise SystemExit(f"pairs needs >=2 symbols on disk; found {len(frames)}")
    prices = align_prices(frames)
    test_start = pd.to_datetime(f"{test_pairs[0][0]}-{test_pairs[0][1]:02d}-01")
    weights = build_weights(prices, formation_days, k, entry, signal, test_start=test_start)
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
    parser = argparse.ArgumentParser(description="Pairs / statistical arbitrage (published-anomaly battery)")
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
