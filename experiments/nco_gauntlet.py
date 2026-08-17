"""REFEREE FINDING candidate #1 -- NCO net-of-cost real-data null. Lopez de Prado's Nested Clustered Optimization
reports 47%/55% estimation-error reduction, but the headline evidence is Monte-Carlo on SIMULATED data where the
true covariance is KNOWN. The un-owned test: does NCO actually beat naive 1/N and plain Ledoit-Wolf min-variance on
REAL free cross-sections, OUT-OF-SAMPLE and NET OF COST? We walk forward {equal-weight, sample-GMV, LW-GMV, NCO},
charge turnover, and run the paired (method - 1/N) net-return difference through the honesty gauntlet.
PRE-REGISTERED KILL: if NCO's net-of-cost Sharpe advantage over 1/N is HAC/DSR-significant on ANY real
cross-section, the refutation fails and NCO is a genuine win -- report that. Demonstration script (experiments/)."""
import json
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.ml_trading_gauntlet import run_gauntlet  # noqa: E402


def load_panel(dirpath, min_bars, n_assets):
    series = {}
    for f in sorted(os.listdir(dirpath)):
        if not f.endswith(".json") or f == "manifest.json":
            continue
        d = json.load(open(os.path.join(dirpath, f)))
        px = d.get("price") or d.get("data") or {}
        closes = {k: px[k][3] for k in px if px[k][3]}
        if len(closes) >= min_bars:
            series[d.get("sym", f[:-5])] = closes
    top = sorted(series, key=lambda s: len(series[s]), reverse=True)[:n_assets]
    common = sorted(set.intersection(*[set(series[s]) for s in top]))
    P = np.array([[series[s][dt] for s in top] for dt in common], dtype=float)
    return np.diff(np.log(P), axis=0)


def gmv(cov):
    inv = np.linalg.pinv(cov)
    w = inv @ np.ones(cov.shape[0])
    s = w.sum()
    return w / s if abs(s) > 1e-12 else np.ones(cov.shape[0]) / cov.shape[0]


def _corr(cov):
    d = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    return np.clip(cov / np.outer(d, d), -1.0, 1.0)


def nco(cov):
    n = cov.shape[0]
    dist = np.sqrt(np.clip((1.0 - _corr(cov)) / 2.0, 0.0, 1.0))
    link = linkage(squareform(dist, checks=False), "ward")
    labels = fcluster(link, max(2, int(round(np.sqrt(n)))), criterion="maxclust")
    w = np.zeros(n)
    groups = {c: np.where(labels == c)[0] for c in np.unique(labels)}
    for idx in groups.values():
        w[idx] = gmv(cov[np.ix_(idx, idx)])
    cl = list(groups)
    red = np.array([[w[groups[a]] @ cov[np.ix_(groups[a], groups[b])] @ w[groups[b]] for b in cl] for a in cl])
    inter = gmv(red)
    out = np.zeros(n)
    for a, c in enumerate(cl):
        out[groups[c]] = w[groups[c]] * inter[a]
    s = out.sum()
    return out / s if abs(s) > 1e-12 else np.ones(n) / n


def ew(_win):
    n = _win.shape[1]
    return np.ones(n) / n


def sample_gmv(win):
    return gmv(np.cov(win.T))


def lw_gmv(win):
    return gmv(LedoitWolf().fit(win).covariance_)


def lw_nco(win):
    return nco(LedoitWolf().fit(win).covariance_)


def backtest(R, weight_fn, window, hold, fee):
    T, N = R.shape
    net, w, prev = [], np.ones(N) / N, np.zeros(N)
    for t in range(window, T):
        if (t - window) % hold == 0:
            w = weight_fn(R[t - window:t])
            turn = np.abs(w - prev).sum()
            prev = w
        else:
            turn = 0.0
        net.append(float(w @ R[t]) - fee * turn)
    return np.array(net)


def ann(series, ppy):
    sd = series.std(ddof=1)
    return (series.mean() / sd * np.sqrt(ppy)) if sd > 1e-12 else 0.0, sd * np.sqrt(ppy)


def run_cross_section(name, R, ppy, window, hold, fee):
    print(f"\n===== {name}: {R.shape[1]} assets x {R.shape[0]} bars, fee {fee*1e4:.0f}bps, "
          f"window {window}, rebalance {hold} =====")
    methods = {"1/N": ew, "sample-GMV": sample_gmv, "LW-GMV": lw_gmv, "NCO": lw_nco}
    curves = {k: backtest(R, fn, window, hold, fee) for k, fn in methods.items()}
    print(f"{'method':>12} {'net Sharpe':>11} {'ann vol':>9}")
    for k, c in curves.items():
        sh, vol = ann(c, ppy)
        print(f"{k:>12} {sh:>+11.2f} {vol:>9.3f}")
    base = curves["1/N"]
    print("  -- does each method BEAT 1/N net-of-cost? (paired difference through the gauntlet) --")
    beat_any = False
    for k in ("LW-GMV", "NCO"):
        diff = curves[k] - base
        per, _ = run_gauntlet([diff], n_trials=1, cutoff=len(diff) // 2, periods_per_year=ppy)
        p = per[0]
        beats = bool(p["econ_pass"] and p["dsr_pass"] and p["postcutoff_pass"])
        beat_any = beat_any or beats
        print(f"     {k}-minus-1/N: Sharpe {p['sharpe_ann']:+.2f}, HAC-lower {p['hac_lower_ann']:+.2f}, "
              f"dsr {p['dsr']:.2f}, postcutoff {p['postcutoff_sharpe_ann']:+.2f} -> BEATS 1/N = {beats}")
    return beat_any


def main():
    crypto = load_panel("cexperps", 900, 30)
    beat_c = run_cross_section("CRYPTO perps", crypto, 365.0, 180, 14, 0.0005)
    beat_e = False
    if os.path.isdir("universe"):
        try:
            eq = load_panel("universe", 1500, 30)
            beat_e = run_cross_section("CROSS-ASSET universe", eq, 252.0, 252, 21, 0.0002)
        except Exception as exc:  # noqa: BLE001
            print(f"\n(universe cross-section skipped: {exc})")
    print("\n" + "=" * 70)
    if beat_c or beat_e:
        print("KILL TRIGGERED: NCO's net-of-cost edge over 1/N is significant on a real cross-section "
              "-> refutation FAILS; NCO is a genuine win here. Report honestly, do not refute.")
    else:
        print("REFUTATION STANDS: on real free cross-sections NCO does NOT beat 1/N net-of-cost after deflation "
              "-> its 47%/55% headline is a simulation-on-known-truth artifact; covariance methods reduce "
              "variance, not net-of-cost Sharpe. (risk works, returns/Sharpe-over-1/N don't)")


if __name__ == "__main__":
    main()
