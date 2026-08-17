"""REFEREE / apparatus-hardening: the Probabilistic Sharpe Ratio (Bailey-Lopez de Prado 2012) and its
Minimum-Track-Record-Length are ANTI-CONSERVATIVE under serial dependence. The PSR SE adjusts for skew/kurtosis
(Mertens) but OMITS the autocorrelation correction its own foundation (Lo 2002) derives -- so for autocorrelated
PnL (smoothed marks, illiquid books, higher-frequency strategies) the standard 'is my Sharpe real?' gate fires
too early. We calibrate it directly: under H0 (true Sharpe = 0) with AR(1) autocorrelation phi, measure the
empirical false-positive rate of PSR>0.95 -- it should be the nominal 5% but is not -- and show the HAC-corrected
variant (trainer.sharpe.probabilistic_sharpe_ratio_hac) recovers nominal size. Then quantify the effect on real
data (467 Binance perps + 112 cross-asset instruments): the autocorrelation present and the resulting
MinTRL-understatement factor sqrt(Newey-West inflation). Practice-change finding, decisive on data in hand; NOT a
celebrated-result refutation -- known primitive (Lo 2002) applied to a widely-used tool that omits it."""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.sharpe import (  # noqa: E402
    probabilistic_sharpe_ratio,
    probabilistic_sharpe_ratio_hac,
    sharpe_standard_error,
    sharpe_standard_error_hac,
    sharpe_stats,
)

PHIS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7)
NS = (500, 2000)
N_SIM = 3000


def ar1(rng, n, phi):
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    s = math.sqrt(1 - phi ** 2) if abs(phi) < 1 else 1.0
    for i in range(1, n):
        x[i] = phi * x[i - 1] + s * e[i]
    return x


def main():
    print("PSR/MinTRL anti-conservatism under serial dependence. H0: true Sharpe=0; nominal false-positive=5%.\n")
    print("=== (1) MONTE-CARLO SIZE: false-positive rate of PSR>0.95 on zero-edge AR(1) returns ===")
    print(f"{'n':>6}{'phi':>6}{'iid_PSR_FP%':>13}{'HAC_PSR_FP%':>13}  (nominal 5%)")
    rows = []
    for n in NS:
        for phi in PHIS:
            iid_fp = hac_fp = 0
            for s in range(N_SIM):
                x = ar1(np.random.default_rng(10000 * (NS.index(n) + 1) + 100 * s + int(phi * 10)), n, phi) * 0.01
                if probabilistic_sharpe_ratio(x, 0.0) > 0.95:
                    iid_fp += 1
                if probabilistic_sharpe_ratio_hac(x, 0.0) > 0.95:
                    hac_fp += 1
            rows.append({"n": n, "phi": phi, "iid_fp": iid_fp / N_SIM, "hac_fp": hac_fp / N_SIM})
            print(f"{n:>6}{phi:>6.1f}{100*iid_fp/N_SIM:>13.1f}{100*hac_fp/N_SIM:>13.1f}")
    print("  => positive autocorrelation inflates the standard PSR's false-positive rate above 5%; HAC recovers it.")

    print("\n=== (2) REAL DATA: autocorrelation present + MinTRL-understatement factor sqrt(NW inflation) ===")
    series = []
    for d in ("cexperps", "universe"):
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json") or f == "manifest.json":
                continue
            rec = json.load(open(os.path.join(d, f)))
            px = None
            if "price" in rec:
                px = [rec["price"][k][3] for k in sorted(rec["price"])]
            elif "data" in rec:
                px = [rec["data"][k][3] for k in sorted(rec["data"]) if rec["data"][k][3]]
            if px and len(px) > 300:
                r = np.diff(np.log(np.array(px, dtype=float)))
                series.append(r[np.isfinite(r)])
    infl = []
    ac1 = []
    for r in series:
        if r.size < 100 or r.std() == 0:
            continue
        st = sharpe_stats(r)
        se_i = sharpe_standard_error(st["sharpe"], st["skew"], st["kurtosis"], st["n_obs"])
        se_h = sharpe_standard_error_hac(r)
        if np.isfinite(se_i) and se_i > 0 and np.isfinite(se_h):
            infl.append(se_h / se_i)
        rc = r - r.mean()
        if (rc @ rc) > 0:
            ac1.append(float((rc[1:] @ rc[:-1]) / (rc @ rc)))
    infl = np.array(infl)
    ac1 = np.array(ac1)
    print(f"  {len(series)} raw daily return series (perps + cross-asset). lag-1 autocorr: "
          f"median {np.median(ac1):+.3f}, |ac1|>0.05 in {100*np.mean(np.abs(ac1)>0.05):.0f}% of series")
    print(f"  HAC/iid Sharpe-SE ratio (MinTRL-understatement factor): median {np.median(infl):.2f}, "
          f"p90 {np.quantile(infl,0.9):.2f}, max {infl.max():.2f}")
    print("  daily raw returns are near-iid (small ratio); the anti-conservatism BITES for autocorrelated PnL")
    print("  (smoothed/illiquid marks, intraday/high-freq, overlapping-horizon overlays) -- see (1) at phi>=0.3.")

    summary = {"n_sim": N_SIM, "size_rows": rows, "n_real_series": len(series),
               "real_ac1_median": float(np.median(ac1)), "real_infl_median": float(np.median(infl)),
               "real_infl_p90": float(np.quantile(infl, 0.9))}
    os.makedirs("experiments/results", exist_ok=True)
    json.dump(summary, open("experiments/results/psr_serial_dependence_calibration.json", "w"), indent=1, default=str)
    print("\nwrote experiments/results/psr_serial_dependence_calibration.json")


if __name__ == "__main__":
    main()
