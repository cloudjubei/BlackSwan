"""METHOD bet, corrected core: DE-CONFOUND mutual-information headroom from time-series PERSISTENCE. The v1 proof-
seed failed verification because (a) it was not in a faithful KMZ regime and (b) its returns were i.i.d. so
transaction cost was inert -- and on REAL data high-MI targets (realized vol) are ALSO high-persistence, so MI and
persistence/cost are confounded. This engine dials them INDEPENDENTLY: features follow AR(1) with autocorrelation
phi (persistence knob) and the target is a flat-spectrum random-Fourier signal plus Gaussian noise (MI knob),
faithfully in the KMZ regime (true signal in the RFF span; model sweeps its own bank size P with genuine ridge z
on the sample covariance via trainer.complexity_ladder). Positions form a real time series so turnover -- and thus
net-of-cost value -- varies with BOTH complexity and persistence.

Reports: (1) a faithful-KMZ regime check (KMZ's OOS virtue reproduces at genuine z, correct double-descent stat);
(2) the DE-CONFOUNDING MAP -- net-of-cost complexity premium over (MI x phi), separating whether MI headroom or
persistence/cost drives 'complexity pays'; (3) an independent (KSG/binning) MI-axis check. Honest goal: a modest
empirical law, NOT a war-ender."""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.complexity_ladder import kmz_ridge_fit_predict, random_fourier_features  # noqa: E402
from trainer.mutual_information import binning_mi, ksg_mi  # noqa: E402
from trainer.sharpe import sharpe_stats  # noqa: E402

D = 5
P_TRUE = 2000
N_TR = 1200
N_TE = 1200
P_SWEEP = (5, 25, 100, 250, 600, 1500, 4000)
GAMMA = 0.5
SEEDS = 12
ANN = math.sqrt(252.0)
VOL = 0.01
COST_BPS = 5.0
SIMPLE_P = 5
COMPLEX_P = 600
Z_HEADLINE = 1.0
Z_SWEEP = (1e-8, 1e-2, 1.0, 100.0)
MI_RUNGS = (0.01, 0.05, 0.20)
PHIS = (0.0, 0.9, 0.99)


def draw_bank(rng, p):
    return rng.standard_normal((D, p)) * math.sqrt(GAMMA), rng.uniform(0, 2 * math.pi, p)


def ar1_path(rng, n, phi):
    x = np.empty((n, D))
    x[0] = rng.standard_normal(D)
    s = math.sqrt(1.0 - phi ** 2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + s * rng.standard_normal(D)
    return x


def var_s_reference():
    rng = np.random.default_rng(11)
    X = ar1_path(rng, 20000, 0.0)
    Wt, bt = draw_bank(rng, P_TRUE)
    beta = rng.standard_normal(P_TRUE) / math.sqrt(P_TRUE)
    s = random_fourier_features(X, Wt, bt) @ beta
    return float(s.var())


def make_series(rng, n, phi, sigma):
    X = ar1_path(rng, n, phi)
    Wt, bt = draw_bank(rng, P_TRUE)
    beta = rng.standard_normal(P_TRUE) / math.sqrt(P_TRUE)
    s = random_fourier_features(X, Wt, bt) @ beta
    y = s + sigma * rng.standard_normal(n)
    y = y * (VOL / y.std())
    return X, s, y


def net_sharpe(pred_tr, pred_te, y_te, cost_bps):
    sd = pred_tr.std()
    if sd == 0:
        return 0.0, 0.0
    pos = np.clip(pred_te / sd, -3.0, 3.0)
    gross = pos * y_te
    turn = np.abs(np.diff(pos, prepend=pos[0]))
    net = gross - cost_bps * 1e-4 * turn
    st = sharpe_stats(net)
    return st["sharpe"] * ANN, float(turn.mean())


def ic(a, b):
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run_cell(mi, phi, sigma, p_list, z, cost_bps):
    oos_ic = {p: [] for p in p_list}
    net = {p: [] for p in p_list}
    turn = {p: [] for p in p_list}
    for seed in range(SEEDS):
        rng = np.random.default_rng(3000 + seed)
        X, s, y = make_series(rng, N_TR + N_TE, phi, sigma)
        Xtr, ytr = X[:N_TR], y[:N_TR]
        Xte, yte = X[N_TR:], y[N_TR:]
        ytr_c = ytr - ytr.mean()
        for p in p_list:
            Wm, bm = draw_bank(rng, p)
            Ztr, Zte = random_fourier_features(Xtr, Wm, bm), random_fourier_features(Xte, Wm, bm)
            pred_tr, pred_te = kmz_ridge_fit_predict(Ztr, ytr_c, Zte, z)
            oos_ic[p].append(ic(pred_te, yte))
            sh, tn = net_sharpe(pred_tr, pred_te, yte, cost_bps)
            net[p].append(sh)
            turn[p].append(tn)
    return ({p: float(np.mean(oos_ic[p])) for p in p_list},
            {p: np.array(net[p]) for p in p_list},
            {p: float(np.mean(turn[p])) for p in p_list})


def main():
    var_s = var_s_reference()
    sig = {mi: math.sqrt(var_s / (math.exp(2 * mi) - 1.0)) for mi in MI_RUNGS}
    print(f"faithful-KMZ + AR(1) persistence. D={D}, P_true={P_TRUE}, N_tr/te={N_TR}/{N_TE}, "
          f"cost={COST_BPS}bps, Var(s)={var_s:.2e} (rescaled to {VOL:.0%} vol)\n")

    print(f"=== (1) REGIME CHECK: faithful-KMZ OOS virtue at genuine ridge z (MI=0.20, phi=0.9) ===")
    print(f"{'z':>9}  " + "".join(f"{p/N_TR:>8.2f}" for p in P_SWEEP) + "   optC  dd_from_dip")
    for z in Z_SWEEP:
        icd, netd, _ = run_cell(0.20, 0.9, sig[0.20], P_SWEEP, z, COST_BPS)
        vals = [icd[p] for p in P_SWEEP]
        dip_i = int(np.argmin([abs(p / N_TR - 1.0) for p in P_SWEEP]))
        rises = vals[-1] > vals[dip_i] + 0.003
        optc = P_SWEEP[int(np.argmax([netd[p].mean() for p in P_SWEEP]))] / N_TR
        print(f"{z:>9.0e}  " + "".join(f"{v:>8.3f}" for v in vals) + f"   {optc:>4.1f}  {'RISES' if rises else 'flat'}")

    print(f"\n=== (2) DE-CONFOUNDING MAP: net-of-cost complexity premium = Sharpe(P={COMPLEX_P}) - Sharpe(P={SIMPLE_P}) ===")
    print(f"(paired over {SEEDS} seeds, z={Z_HEADLINE}, cost={COST_BPS}bps; * = 95% CI clears 0)")
    print(f"{'MI vs phi':>10}  " + "".join(f"{('phi='+str(ph)):>16}" for ph in PHIS))
    premium_map = {}
    turn_map = {}
    optc_map = {}
    for mi in MI_RUNGS:
        cells = []
        for phi in PHIS:
            _, netd, turnd = run_cell(mi, phi, sig[mi], P_SWEEP, Z_HEADLINE, COST_BPS)
            diff = netd[COMPLEX_P] - netd[SIMPLE_P]
            m = float(diff.mean())
            lo = m - 1.96 * float(diff.std(ddof=1) / math.sqrt(SEEDS))
            star = "*" if lo > 0 else " "
            optc = P_SWEEP[int(np.argmax([netd[p].mean() for p in P_SWEEP]))] / N_TR
            premium_map[(mi, phi)] = m
            turn_map[(mi, phi)] = (turnd[SIMPLE_P], turnd[COMPLEX_P])
            optc_map[(mi, phi)] = optc
            cells.append(f"{m:+6.2f}{star} oc{optc:>4.1f}")
        print(f"{mi:>10.3f}  " + "".join(f"{c:>16}" for c in cells))

    print(f"\n  turnover(simple,complex) per step -- persistence lowers it, complexity raises it:")
    for mi in (MI_RUNGS[-1],):
        for phi in PHIS:
            ts, tc = turn_map[(mi, phi)]
            print(f"    MI={mi} phi={phi}: simple={ts:.3f}  complex={tc:.3f}")

    print(f"\n=== (3) MI-axis check (INDEPENDENT estimators on (signal, target); avoid circular gaussian) ===")
    print(f"{'MI_oracle':>10}{'ksg':>10}{'binning':>10}")
    mi_check = {}
    rngm = np.random.default_rng(99)
    for mi in MI_RUNGS:
        _, s, y = make_series(rngm, 8000, 0.0, sig[mi])
        k, bnn = ksg_mi(s, y, k=5), binning_mi(s, y)
        mi_check[mi] = {"ksg": k, "binning": bnn}
        print(f"{mi:>10.3f}{k:>10.3f}{bnn:>10.3f}")

    summary = {"var_s": var_s, "cost_bps": COST_BPS, "z_headline": Z_HEADLINE, "seeds": SEEDS,
               "premium_map": {f"{mi}|{phi}": premium_map[(mi, phi)] for mi in MI_RUNGS for phi in PHIS},
               "optc_map": {f"{mi}|{phi}": optc_map[(mi, phi)] for mi in MI_RUNGS for phi in PHIS},
               "turn_map": {f"{mi}|{phi}": turn_map[(mi, phi)] for mi in MI_RUNGS for phi in PHIS},
               "mi_check": {str(mi): mi_check[mi] for mi in MI_RUNGS}}
    os.makedirs("experiments/results", exist_ok=True)
    json.dump(summary, open("experiments/results/complexity_mi_persistence.json", "w"), indent=1, default=str)
    print("\nwrote experiments/results/complexity_mi_persistence.json")


if __name__ == "__main__":
    main()
