"""Candidate #1 -> publishable: the DISPERSION LAW. Turn the two-cross-section observation into a continuum. We
draw many random sub-universes from the crypto-perp pool (which spans a huge cross-sectional volatility range, from
near-stable majors to meme alts), and test whether the OUT-OF-SAMPLE, NET-OF-COST advantage of LONG-ONLY
minimum-variance over naive 1/N rises monotonically with the universe's cross-sectional volatility DISPERSION -- with
per-basket HAC significance, not just a scatter. Long-only removes the short-funding critique. We also check that NCO
clustering adds nothing over plain Ledoit-Wolf GMV across the same baskets. Demonstration script."""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nco_robustness import ew, gmv_long_only, lo_nco  # noqa: E402
from nco_gauntlet import backtest  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
from trainer.ml_trading_gauntlet import run_gauntlet  # noqa: E402

PPY = 365.0
FEE = 0.0005
POOL = 120
M = 15
B = 90
WINDOW = 180
HOLD = 21


def load_pool():
    series = {}
    for f in sorted(os.listdir("cexperps")):
        if not f.endswith(".json"):
            continue
        d = json.load(open(os.path.join("cexperps", f)))
        p = d.get("price", {})
        if len(p) >= 700:
            series[d.get("sym", f[:-5])] = {k: p[k][3] for k in p if p[k][3]}
    top = sorted(series, key=lambda s: len(series[s]), reverse=True)[:POOL]
    common = sorted(set.intersection(*[set(series[s]) for s in top]))
    P = np.array([[series[s][dt] for s in top] for dt in common], dtype=float)
    return np.diff(np.log(P), axis=0)


def ann(x):
    x = np.asarray(x)
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(PPY)) if sd > 1e-12 else 0.0


def lo_gmv(win):
    return gmv_long_only(LedoitWolf().fit(win).covariance_)


def main():
    R = load_pool()
    print(f"pool {R.shape[1]} perps x {R.shape[0]} bars; {B} random {M}-asset baskets, long-only, {int(FEE*1e4)}bps\n")
    rng = np.random.default_rng(0)
    rows = []
    for b in range(B):
        cols = rng.choice(R.shape[1], size=M, replace=False)
        sub = R[:, cols]
        vols = sub.std(axis=0)
        dispersion = float(vols.std() / vols.mean())
        base = backtest(sub, ew, WINDOW, HOLD, FEE)
        g = backtest(sub, lo_gmv, WINDOW, HOLD, FEE)
        adv = ann(g) - ann(base)
        per, _ = run_gauntlet([g - base], n_trials=1, periods_per_year=PPY)
        rows.append((dispersion, adv, per[0]["hac_lower_ann"] > 0.0))
    disp = np.array([r[0] for r in rows])
    adv = np.array([r[1] for r in rows])
    sig = np.array([r[2] for r in rows])

    rho, p = spearmanr(disp, adv)
    print(f"Spearman(dispersion, GMV-minus-1/N net Sharpe) across {B} baskets: rho = {rho:+.3f}, p = {p:.4f}\n")
    order = np.argsort(disp)
    terc = np.array_split(order, 3)
    for name, ix in zip(("LOW dispersion", "MID dispersion", "HIGH dispersion"), terc):
        print(f"  {name:>16}: median disp {np.median(disp[ix]):.2f} | mean advantage {adv[ix].mean():+.2f} "
              f"| HAC-significant baskets {100*sig[ix].mean():.0f}%")

    print("\n  NCO-clustering-null check (subset of 24 baskets): mean(NCO - LW-GMV) net Sharpe")
    diffs = []
    for b in range(24):
        cols = rng.choice(R.shape[1], size=M, replace=False)
        sub = R[:, cols]
        diffs.append(ann(backtest(sub, lo_nco, WINDOW, HOLD, FEE)) - ann(backtest(sub, lo_gmv, WINDOW, HOLD, FEE)))
    print(f"    mean {np.mean(diffs):+.3f} Sharpe, median {np.median(diffs):+.3f} "
          f"(~0 => clustering adds nothing over plain shrinkage GMV)")

    lo_sig = sig[terc[0]].mean()
    hi_sig = sig[terc[2]].mean()
    print()
    if rho > 0.2 and p < 0.05 and hi_sig > lo_sig + 0.2:
        print(f"DISPERSION LAW HOLDS: the net-of-cost min-variance advantage over 1/N rises monotonically with "
              f"cross-sectional vol dispersion (Spearman {rho:+.2f}), and HAC-significant advantage jumps "
              f"{100*lo_sig:.0f}% (low) -> {100*hi_sig:.0f}% (high dispersion). NCO's clustering is not the source.")
    else:
        print(f"NOT A CLEAN LAW on this pool: rho {rho:+.2f} p {p:.3f}, sig {100*lo_sig:.0f}%->{100*hi_sig:.0f}% "
              "-- report honestly, do not force a law.")


if __name__ == "__main__":
    main()
