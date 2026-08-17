"""POSITIVE CONTROL for the ML-trading honesty gauntlet, on the free Binance-perp cross-section we already hold:
demonstrate the information/effective-trials boundary directly -- (1) the INFORMATION side: next-period realized
VOLATILITY carries far more mutual information about its own past than next-period signed RETURN does (risk is
predictable, direction is not), and (2) the ECONOMIC side: a canonical cross-sectional RETURN-timing 'alpha'
(short-term reversal) that looks fine gross does NOT survive the gauntlet net of cost -- its Sharpe fails to beat
matched-complexity random formulas. Together: 'risk works, returns don't', on data in hand, with our own tools.
This is a demonstration script (experiments/), not a unit test."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.ml_trading_gauntlet import run_gauntlet  # noqa: E402
from trainer.mutual_information import ksg_mi  # noqa: E402
from trainer.random_formula_null import (  # noqa: E402
    random_formula_null,
    signal_to_weights,
)
from trainer.trading_costs import net_return_series  # noqa: E402

PPY = 365.0
FEE = 0.0005
N_ASSETS = 40
MIN_BARS = 900


def load_closes(dirpath):
    out = {}
    for f in sorted(os.listdir(dirpath)):
        if not f.endswith(".json") or f == "manifest.json":
            continue
        d = json.load(open(os.path.join(dirpath, f)))
        px = d.get("price", {})
        if len(px) < MIN_BARS:
            continue
        closes = {dt: px[dt][3] for dt in px if px[dt][3]}
        out[d.get("sym", f[:-5])] = closes
    return out


def rolling_std(x, k):
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        out[i] = x[max(0, i - k + 1):i + 1].std(axis=0)
    return out


def ann_sharpe(r):
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(PPY)) if sd > 1e-12 else 0.0


def main():
    series = load_closes("cexperps")
    top = sorted(series, key=lambda s: len(series[s]), reverse=True)[:N_ASSETS]
    common = sorted(set.intersection(*[set(series[s]) for s in top]))
    P = np.array([[series[s][dt] for s in top] for dt in common], dtype=float)
    R = np.diff(np.log(P), axis=0)
    RV = rolling_std(R, 5)
    R, RV = R[5:], RV[5:]
    T, N = R.shape
    print(f"panel: {N} perps x {T} daily bars (common range {common[6]}..{common[-1]})\n")

    mi_ret = np.median([ksg_mi(R[:-1, i], R[1:, i]) for i in range(N)])
    mi_vol = np.median([ksg_mi(RV[:-1, i], RV[1:, i]) for i in range(N)])
    print("=== INFORMATION side (median per-asset mutual information, nats) ===")
    print(f"  MI(past return -> next return)         = {mi_ret:.4f}")
    print(f"  MI(past realized-vol -> next real-vol) = {mi_vol:.4f}")
    print(f"  ratio vol/return = {mi_vol / mi_ret:.1f}x  -> risk is predictable, direction is ~not\n")

    weights = signal_to_weights(-R)
    lagged = np.vstack([np.zeros((1, N)), weights[:-1]])
    strat_gross = net_return_series(lagged, R, 0.0)
    strat_net = net_return_series(lagged, R, FEE)
    print("=== ECONOMIC side: cross-sectional 1-day REVERSAL (the return-timing alpha) ===")
    print(f"  gross annualised Sharpe = {ann_sharpe(strat_gross):+.2f}")
    print(f"  net   annualised Sharpe = {ann_sharpe(strat_net):+.2f}  (taker fee {FEE*1e4:.0f} bps on turnover)")

    features = [R, RV, np.sign(R)]
    null = random_formula_null(np.random.default_rng(0), features, R, depth=2, k=300, fee=FEE, periods_per_year=PPY)
    per, fam = run_gauntlet([strat_net], n_trials=1, null_sharpes=null, cutoff=T // 2, periods_per_year=PPY)
    p = per[0]
    print(f"  matched-complexity random-formula null: median Sharpe {np.median(null):+.2f}, "
          f"95th pct {np.quantile(null, 0.95):+.2f}")
    print(f"  gauntlet: null_p={p['null_p']:.3f} null_pass={p['null_pass']} | econ_pass={p['econ_pass']} | "
          f"dsr_pass={p['dsr_pass']} | postcutoff_pass={p['postcutoff_pass']}")
    print(f"  -> SURVIVES = {p['survives']}\n")

    verdict = ("returns-don't confirmed: reversal is not a net-of-cost discovery"
               if not p["survives"] else "reversal SURVIVED net-of-cost -- report as a genuine edge, do not refute")
    print(f"VERDICT: MI(vol) >> MI(return) and {verdict}.")


if __name__ == "__main__":
    main()
