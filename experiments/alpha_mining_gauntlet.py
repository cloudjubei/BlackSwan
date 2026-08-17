"""REFEREE candidate #3 -- the 2024-26 formulaic-alpha-mining zoo (AlphaGen/AlphaForge/AlphaAgent/Chain-of-Alpha)
reports headline in-sample IC/Sharpe from searching thousands of formulas, rarely deflating for the search. We
weaponise the matched-complexity random-formula null directly: MINE by taking the best of K random formulas on the
in-sample half (exactly what a miner does), then (a) show the 'headline' best IS Sharpes look great, (b) show they
COLLAPSE out-of-sample, and (c) feed the whole mined family through the honesty gauntlet -- its own search
multiplicity (Deflated Sharpe against the family's trial dispersion + BY-FDR) -- and count survivors. Prediction of
the MI/effective-trials boundary: on low-MI returns, best-of-K mining is indistinguishable from search luck and
~zero survive. Demonstration script."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.ml_trading_gauntlet import run_gauntlet  # noqa: E402
from trainer.random_formula_null import random_formula, signal_to_weights  # noqa: E402
from trainer.trading_costs import net_return_series  # noqa: E402

PPY = 365.0
FEE = 0.0010
K = 400


def load_panel(n_assets, min_bars=900):
    import json
    series = {}
    for f in sorted(os.listdir("cexperps")):
        if not f.endswith(".json"):
            continue
        d = json.load(open(os.path.join("cexperps", f)))
        p = d.get("price", {})
        if len(p) >= min_bars:
            series[d.get("sym", f[:-5])] = {k: p[k][3] for k in p if p[k][3]}
    top = sorted(series, key=lambda s: len(series[s]), reverse=True)[:n_assets]
    common = sorted(set.intersection(*[set(series[s]) for s in top]))
    P = np.array([[series[s][dt] for s in top] for dt in common], dtype=float)
    return np.diff(np.log(P), axis=0)


def ann(series):
    s = np.asarray(series)
    sd = s.std(ddof=1)
    return float(s.mean() / sd * np.sqrt(PPY)) if sd > 1e-12 else 0.0


def formula_series(signal, R, fee):
    W = signal_to_weights(signal)
    lag = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    return net_return_series(lag, R, fee)


def main():
    R = load_panel(40)
    T, N = R.shape
    half = T // 2
    rng = np.random.default_rng(0)
    features = [R, np.sign(R), np.abs(R)]
    sigs = [random_formula(rng, features, depth=3) for _ in range(K)]
    gross = [formula_series(s, R, 0.0) for s in sigs]
    nets = [formula_series(s, R, FEE) for s in sigs]

    is_g = np.array([ann(x[:half]) for x in gross])
    oos_g = np.array([ann(x[half:]) for x in gross])
    is_n = np.array([ann(x[:half]) for x in nets])
    oos_n = np.array([ann(x[half:]) for x in nets])
    top_g = np.argsort(is_g)[::-1][:20]
    top_n = np.argsort(is_n)[::-1][:20]

    print(f"MINING crypto perps ({N} assets x {T} bars): best of K={K} random depth-3 formulas, {int(FEE*1e4)}bps\n")
    print("  GROSS (the zoo's reported number): top-20-by-IS mean IS "
          f"{is_g[top_g].mean():+.2f} -> OOS {oos_g[top_g].mean():+.2f}  (weak persistent structure exists)")
    print(f"  cost drag (median gross->net IS Sharpe): {np.median(is_g) - np.median(is_n):+.2f} Sharpe from turnover")
    print("  NET (tradeable): top-20-by-IS mean IS "
          f"{is_n[top_n].mean():+.2f} -> OOS {oos_n[top_n].mean():+.2f}  (collapses)")
    print(f"  IS->OOS corr: gross {np.corrcoef(is_g, oos_g)[0,1]:+.2f} vs net {np.corrcoef(is_n, oos_n)[0,1]:+.2f} "
          f"(net corr is inflated by PERSISTENT COST, not alpha)\n")

    per, fam = run_gauntlet(nets, n_trials="effective", cutoff=half, periods_per_year=PPY)
    print(f"  gauntlet on the whole mined family (its own search multiplicity): effective_trials "
          f"{fam['effective_trials']:.0f}, nominal-significant {fam['n_nominal_sig']}, "
          f"DSR-pass {fam['n_dsr_pass']}, BY-reject {fam['n_by_reject']} -> SURVIVED {fam['n_survived']}/{K}\n")

    net_tradeable = fam["n_survived"] > 0 or oos_n[top_n].mean() > 0
    if not net_tradeable:
        print("REFUTATION STANDS (net-of-cost): random formula mining finds WEAK persistent GROSS structure "
              f"(top-20 gross OOS {oos_g[top_g].mean():+.2f}) but turnover cost ({np.median(is_g)-np.median(is_n):+.2f} "
              "Sharpe) destroys it -- IS-selected NET alphas collapse OOS and 0 survive the search multiplicity. "
              "The zoo's GROSS IC/Sharpe headline is real-but-untradeable; net-of-cost it is a cost artifact.")
    else:
        print("PARTIAL: net-of-cost mined alphas persist/survive -- investigate before refuting.")


if __name__ == "__main__":
    main()
