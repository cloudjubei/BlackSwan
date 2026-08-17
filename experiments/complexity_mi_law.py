"""METHOD-bet proof-seed v1 -- FAILED adversarial verification (`verify-mi-complexity-law`, recordable=false).
SUPERSEDED; kept only as the honest record of what does and does NOT survive. Do not cite the "reconciliation".

What this first cut got WRONG (all verified by an independent faithful-KMZ rebuild, scratchpad/kmz_faithful.py):
  - The "OOS peaks at moderate complexity / double-descent weak-or-absent" headline was a REGIME ARTIFACT of an
    inert ridge (lam*I added to Z'Z of scale ~N => all three "lambdas" are the SAME near-ridgeless fit), an
    unnormalized RFF (sqrt(2) not sqrt(2/P) => effective shrinkage vanishes at high P), c capped at 5, and a
    MIS-SPECIFIED double-descent test (compared post-interpolation to the PRE-interp peak instead of the
    interpolation dip -- the JSON's own OOS-IC actually RISES past interpolation). In a faithful KMZ setup
    (flat spectrum, genuine ridge z on the sample covariance, normalized RFF, dual solve to c=20) KMZ's OOS
    virtue REPRODUCES -- with z=100 the most-complex model is globally optimal (OOS Sharpe rises to ~3.08 at
    c=20). So this cut does NOT refute KMZ; if anything it confirms them in their regime.
  - The transaction-cost axis is INERT: test rows are i.i.d., so turnover is complexity-invariant and cost
    cancels in the paired premium. The "+0.85*..+4.54* net-of-cost" figures are actually the GROSS (cost=0) row;
    "MI* rises with cost" is false.
  - MI* is NOT a structural constant -- it is a power/comparison-point-dependent significance crossing on a
    smooth monotone premium curve (0.05 at complex-c=0.2, 0.2 at complex-c=10, ~0.02 under a different RNG order).
  - The MI axis is a Gaussian UPPER BOUND (s=g(X) is non-Gaussian; true MI is lower); the gaussian/copula
    "cross-check" is circular (algebraically equals the oracle formula). Only large-n KSG/binning are independent.

What SURVIVES (the only recordable claim): gross OOS timing skill is monotonically MI-GATED and LEAKAGE-CLEAN --
the complexity premium (high-P vs degenerate low-P RFF-ridge) sign-flips from ~0 at near-zero MI to strongly
positive at high MI. That is a leakage-clean sanity gate, NOT a law and NOT a reconciliation. Establishing
anything landmark-worthy needs: (1) a faithful-KMZ regime, (2) time-series PERSISTENCE so cost can bite, and
(3) REAL financial targets (realized vol vs daily return) where MI and persistence/cost are confounded."""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.mutual_information import estimate_mi  # noqa: E402
from trainer.sharpe import sharpe_stats  # noqa: E402

N_TRAIN = 500
N_TEST = 4000
D = 6
P_SWEEP = (5, 25, 100, 250, 500, 1000, 2500)
MI_RUNGS = (0.005, 0.02, 0.05, 0.1, 0.2, 0.4)
SEEDS = 16
RFF_GAMMA = 0.5
RIDGE_LAMBDA = 1e-6
LAMBDAS_ROBUST = (1e-6, 1e-3, 1e-1)
MI_PROBE = 0.1
SIMPLE_P = 5
COMPLEX_P = 100
COST_KAPPAS = (0.0, 0.02, 0.05, 0.10)
ANN = math.sqrt(252.0)


def signal(X):
    return (np.tanh(X[:, 0] * X[:, 1] + X[:, 2] ** 2 - X[:, 3])
            + 0.5 * np.sin(2.0 * X[:, 4]) + 0.5 * X[:, 5] * X[:, 0])


def make_target(rng, n, sigma, s_scale):
    X = rng.standard_normal((n, D))
    s = signal(X) * s_scale
    y = s + sigma * rng.standard_normal(n)
    return X, s, y


def rff(X, W, b):
    return np.cos(X @ W + b) * math.sqrt(2.0)


def ridge_fit_predict(Ztr, ytr, Zte, lam=RIDGE_LAMBDA):
    p = Ztr.shape[1]
    A = Ztr.T @ Ztr + lam * np.eye(p)
    w = np.linalg.solve(A, Ztr.T @ ytr)
    return Ztr @ w, Zte @ w


def ic(pred, actual):
    if pred.std() == 0 or actual.std() == 0:
        return 0.0
    return float(np.corrcoef(pred, actual)[0, 1])


def timing_net_sharpe(pred, y, cost_rate):
    sd = pred.std()
    if sd == 0:
        return 0.0
    pos = np.clip(pred / sd, -3.0, 3.0)
    gross = pos * y
    turn = np.abs(np.diff(pos, prepend=0.0))
    net = gross - cost_rate * turn
    st = sharpe_stats(net)
    return st["sharpe"] * ANN


def sigma_for_mi(mi, var_s):
    return math.sqrt(var_s / (math.exp(2.0 * mi) - 1.0))


def probe_lambda(var_s_unit):
    sigma = sigma_for_mi(MI_PROBE, var_s_unit)
    out = {lam: {p: [] for p in P_SWEEP} for lam in LAMBDAS_ROBUST}
    for seed in range(SEEDS):
        rng = np.random.default_rng(5000 + seed)
        Xtr, _, ytr = make_target(rng, N_TRAIN, sigma, 1.0)
        Xte, _, yte = make_target(rng, N_TEST, sigma, 1.0)
        ytr_c = ytr - ytr.mean()
        for p in P_SWEEP:
            Wp = rng.standard_normal((D, p)) * math.sqrt(RFF_GAMMA)
            bp = rng.uniform(0, 2 * math.pi, p)
            Ztr, Zte = rff(Xtr, Wp, bp), rff(Xte, Wp, bp)
            for lam in LAMBDAS_ROBUST:
                _, pred_te = ridge_fit_predict(Ztr, ytr_c, Zte, lam=lam)
                out[lam][p].append(ic(pred_te, yte))
    return {lam: {p: float(np.mean(out[lam][p])) for p in P_SWEEP} for lam in LAMBDAS_ROBUST}


def main():
    base = np.random.default_rng(0)
    Xb, sb, _ = make_target(base, 20000, 0.0, 1.0)
    var_s_unit = float(sb.var())
    print(f"synthetic KMZ timing: N_train={N_TRAIN} d={D}, RFF ridge, complexity c=P/N in "
          f"{[round(p/N_TRAIN,2) for p in P_SWEEP]}; nonlinear signal Var={var_s_unit:.3f}\n")

    net_by_cost = {k: {} for k in COST_KAPPAS}
    net_raw = {k: {} for k in COST_KAPPAS}
    is_ic = {}
    oos_ic = {}
    mi_check = {}
    for mi in MI_RUNGS:
        sigma = sigma_for_mi(mi, var_s_unit)
        sy = 1.0
        rng0 = np.random.default_rng(777)
        Xc, sc, yc = make_target(rng0, 8000, sigma, sy)
        mi_check[mi] = estimate_mi(sc, yc)
        per_p_isic = {p: [] for p in P_SWEEP}
        per_p_oosic = {p: [] for p in P_SWEEP}
        per_p_net = {p: {k: [] for k in COST_KAPPAS} for p in P_SWEEP}
        for seed in range(SEEDS):
            rng = np.random.default_rng(1000 + seed)
            Xtr, _, ytr = make_target(rng, N_TRAIN, sigma, sy)
            Xte, ste, yte = make_target(rng, N_TEST, sigma, sy)
            ytr_c = ytr - ytr.mean()
            for p in P_SWEEP:
                Wp = rng.standard_normal((D, p)) * math.sqrt(RFF_GAMMA)
                bp = rng.uniform(0, 2 * math.pi, p)
                pred_tr, pred_te = ridge_fit_predict(rff(Xtr, Wp, bp), ytr_c, rff(Xte, Wp, bp))
                per_p_isic[p].append(ic(pred_tr, ytr_c))
                per_p_oosic[p].append(ic(pred_te, yte))
                for k in COST_KAPPAS:
                    per_p_net[p][k].append(timing_net_sharpe(pred_te, yte, k * sy))
        is_ic[mi] = {p: float(np.mean(per_p_isic[p])) for p in P_SWEEP}
        oos_ic[mi] = {p: float(np.mean(per_p_oosic[p])) for p in P_SWEEP}
        for k in COST_KAPPAS:
            net_raw[k][mi] = {p: np.array(per_p_net[p][k]) for p in P_SWEEP}
            net_by_cost[k][mi] = {p: (float(np.mean(per_p_net[p][k])),
                                      float(np.std(per_p_net[p][k], ddof=1) / math.sqrt(SEEDS)))
                                  for p in P_SWEEP}

    print("=== MI-axis validation (estimators on (signal, target) recover the oracle MI budget) ===")
    print(f"{'MI_oracle':>10}  " + "".join(f"{e:>10}" for e in ('gaussian', 'copula', 'binning', 'ksg')))
    for mi in MI_RUNGS:
        e = mi_check[mi]
        print(f"{mi:>10.3f}  " + "".join(f"{e[k]:>10.3f}" for k in ('gaussian', 'copula', 'binning', 'ksg')))

    print("\n=== IN-SAMPLE IC (the KMZ 'virtue': rises with complexity c=P/N at EVERY MI) ===")
    _grid(is_ic)
    print("\n=== OOS IC (the virtue survives only where there is MI to extract) ===")
    _grid(oos_ic)

    for k in COST_KAPPAS:
        print(f"\n=== OOS NET-OF-COST timing Sharpe (cost kappa={k}) -- mean [+/-1.96 SE] ===")
        _grid_net(net_by_cost[k])

    print(f"\n=== lambda-ROBUSTNESS at MI={MI_PROBE} (does ridgeless show double-descent? KMZ vs Nagel crux) ===")
    print(f"OOS IC by complexity, per ridge lambda; c>1 is PAST interpolation")
    print(f"{'lambda':>9}  " + "".join(f"{p/N_TRAIN:>7.2f}" for p in P_SWEEP))
    dd = probe_lambda(var_s_unit)
    for lam in LAMBDAS_ROBUST:
        row = dd[lam]
        past = "RISES past interp" if row[P_SWEEP[-1]] > row[P_SWEEP[len(P_SWEEP)//2]] + 0.005 else "declines/flat"
        print(f"{lam:>9.0e}  " + "".join(f"{row[p]:>7.3f}" for p in P_SWEEP) + f"  ({past})")

    print(f"\n=== THE LAW: complexity PREMIUM = OOS-net Sharpe(P={COMPLEX_P}) - Sharpe(P={SIMPLE_P}), paired CI ===")
    print(f"MI*_complexity(cost) = smallest MI where the premium's 95% CI clears 0 (complexity PAYS net-of-cost)")
    law = {}
    for k in COST_KAPPAS:
        mistar = None
        line = []
        for mi in MI_RUNGS:
            diff = net_raw[k][mi][COMPLEX_P] - net_raw[k][mi][SIMPLE_P]
            m = float(diff.mean())
            lo = m - 1.96 * float(diff.std(ddof=1) / math.sqrt(SEEDS))
            line.append(f"{mi}:{m:+.2f}{'*' if lo > 0 else ' '}")
            if lo > 0 and mistar is None:
                mistar = mi
        law[k] = mistar
        print(f"  cost k={k:<5} MI*={str(mistar):<6} | premium " + "  ".join(line))
    mono = all((law[COST_KAPPAS[i]] or 9) <= (law[COST_KAPPAS[i + 1]] or 9)
               for i in range(len(COST_KAPPAS) - 1))
    opt_c = {mi: max(P_SWEEP, key=lambda p: net_by_cost[0.05][mi][p][0]) / N_TRAIN for mi in MI_RUNGS}
    print(f"  MI*_complexity monotonically non-decreasing in cost: {mono}")
    print(f"  optimal complexity c=P/N (at cost 0.05) by MI: " + ", ".join(f"{mi}:{opt_c[mi]:.2f}" for mi in MI_RUNGS))

    summary = {"n_train": N_TRAIN, "d": D, "p_sweep": list(P_SWEEP), "mi_rungs": list(MI_RUNGS),
               "seeds": SEEDS, "simple_p": SIMPLE_P, "complex_p": COMPLEX_P,
               "mi_star_complexity_by_cost": {str(k): law[k] for k in COST_KAPPAS},
               "premium_by_cost": {str(k): {str(mi): float((net_raw[k][mi][COMPLEX_P] - net_raw[k][mi][SIMPLE_P]).mean())
                                            for mi in MI_RUNGS} for k in COST_KAPPAS},
               "optimal_c_cost05": {str(mi): opt_c[mi] for mi in MI_RUNGS},
               "oos_ic": {str(mi): oos_ic[mi] for mi in MI_RUNGS},
               "lambda_double_descent": {str(lam): {str(p): dd[lam][p] for p in P_SWEEP} for lam in LAMBDAS_ROBUST},
               "mi_estimator_check": {str(mi): mi_check[mi] for mi in MI_RUNGS}}
    os.makedirs("experiments/results", exist_ok=True)
    json.dump(summary, open("experiments/results/complexity_mi_law.json", "w"), indent=1, default=str)
    print("\nwrote experiments/results/complexity_mi_law.json")


def _grid(table):
    print(f"{'MI|c':>9}  " + "".join(f"{p/N_TRAIN:>8.2f}" for p in P_SWEEP))
    for mi in MI_RUNGS:
        print(f"{mi:>9.3f}  " + "".join(f"{table[mi][p]:>8.3f}" for p in P_SWEEP))


def _grid_net(table):
    print(f"{'MI|c':>9}  " + "".join(f"{p/N_TRAIN:>8.2f}" for p in P_SWEEP))
    for mi in MI_RUNGS:
        cells = []
        for p in P_SWEEP:
            m, se = table[mi][p]
            cells.append(f"{m:>8.2f}")
        print(f"{mi:>9.3f}  " + "".join(cells))


if __name__ == "__main__":
    main()
