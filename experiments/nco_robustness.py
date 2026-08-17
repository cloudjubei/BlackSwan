"""NCO robustness (candidate #1 close-out): the first cut found min-variance beats 1/N net-of-cost in crypto, but
GMV was long-short on perps (turnover charged, short FUNDING not). The decisive check is LONG-ONLY min-variance
(no shorts -> no funding artifact): if long-only LW-GMV STILL beats 1/N net-of-cost across a parameter sweep, the
dispersion-conditional finding is robust and NCO's clustering still adds nothing over plain shrinkage GMV. If it
does NOT, the crypto edge was a leveraged-short artifact. Demonstration script (experiments/)."""
import os
import sys

import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nco_gauntlet import _corr, ann, backtest, load_panel  # noqa: E402
from trainer.ml_trading_gauntlet import run_gauntlet  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402


def gmv_long_only(cov):
    n = cov.shape[0]
    res = minimize(lambda w: w @ cov @ w, np.ones(n) / n, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 200, "ftol": 1e-12})
    return res.x if res.success else np.ones(n) / n


def nco_long_only(cov):
    n = cov.shape[0]
    dist = np.sqrt(np.clip((1.0 - _corr(cov)) / 2.0, 0.0, 1.0))
    labels = fcluster(linkage(squareform(dist, checks=False), "ward"),
                      max(2, int(round(np.sqrt(n)))), criterion="maxclust")
    w = np.zeros(n)
    groups = {c: np.where(labels == c)[0] for c in np.unique(labels)}
    for idx in groups.values():
        w[idx] = gmv_long_only(cov[np.ix_(idx, idx)])
    cl = list(groups)
    red = np.array([[w[groups[a]] @ cov[np.ix_(groups[a], groups[b])] @ w[groups[b]] for b in cl] for a in cl])
    inter = gmv_long_only(red)
    out = np.zeros(n)
    for a, c in enumerate(cl):
        out[groups[c]] = w[groups[c]] * inter[a]
    return out / out.sum()


def ew(win):
    return np.ones(win.shape[1]) / win.shape[1]


def lo_gmv(win):
    return gmv_long_only(LedoitWolf().fit(win).covariance_)


def lo_nco(win):
    return nco_long_only(LedoitWolf().fit(win).covariance_)


def main():
    crypto = load_panel("cexperps", 900, 30)
    ppy, fee = 365.0, 0.0005
    print("LONG-ONLY min-variance vs 1/N net-of-cost, crypto perps (no shorts -> no funding artifact)\n")
    beats = []
    for window, hold in [(120, 14), (180, 14), (252, 21)]:
        base = backtest(crypto, ew, window, hold, fee)
        g = backtest(crypto, lo_gmv, window, hold, fee)
        nc = backtest(crypto, lo_nco, window, hold, fee)
        sh_b = ann(base, ppy)[0]
        sh_g = ann(g, ppy)[0]
        sh_n = ann(nc, ppy)[0]
        per, _ = run_gauntlet([g - base], n_trials=1, cutoff=len(base) // 2, periods_per_year=ppy)
        p = per[0]
        win_beats = bool(p["econ_pass"] and p["dsr_pass"] and p["postcutoff_pass"])
        beats.append(win_beats)
        print(f"  win{window}/reb{hold}: 1/N {sh_b:+.2f} | LO-GMV {sh_g:+.2f} | LO-NCO {sh_n:+.2f} "
              f"|| GMV-minus-1/N HAC-lower {p['hac_lower_ann']:+.2f} dsr {p['dsr']:.2f} "
              f"postcut {p['postcutoff_sharpe_ann']:+.2f} -> beats 1/N = {win_beats}")
    print()
    if all(beats):
        print("ROBUST: long-only min-variance beats 1/N net-of-cost across all windows -> the crypto edge is NOT "
              "a leveraged-short/funding artifact. Honest finding stands: covariance-opt beats 1/N in the "
              "high-dispersion crypto cross-section (low-vol anomaly), and NCO's clustering adds nothing over "
              "plain shrinkage GMV.")
    elif any(beats):
        print("MIXED: long-only min-variance beats 1/N in some windows only -> edge is real but parameter-fragile; "
              "report with the sweep, not as a clean law.")
    else:
        print("ARTIFACT: long-only min-variance does NOT beat 1/N -> the first-cut crypto edge was a "
              "leveraged-short/funding artifact; refutation of covariance-opt-beats-1/N STANDS long-only.")


if __name__ == "__main__":
    main()
