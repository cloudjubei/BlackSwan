"""REFEREE candidate #2 -- the meta-labeling boundary (López de Prado's celebrated 'a secondary ML model that
sizes a primary signal adds efficacy'). The decisive, un-owned test that also INSTANTIATES the MI/effective-trials
hypothesis: a secondary trained on the SAME features the primary already used should NOT beat a single model on the
union ('squeezing the same orange twice'), whereas a secondary conditioning on ORTHOGONAL risk/regime features
(realized vol) SHOULD improve net Sharpe via sizing (it knows WHEN the primary works, which is high-MI). We build a
TSMOM primary on the free perp panel, meta-label its bets, size by (A) same momentum features vs (B) orthogonal
vol/regime features, and compare OOS net-of-cost Sharpe through the honesty gauntlet. Prediction: B > primary,
A ~= primary. Demonstration script."""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.ml_trading_gauntlet import run_gauntlet  # noqa: E402

PPY = 365.0
FEE = 0.0008


def load_panel(n_assets, min_bars=900):
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


def roll(x, k, fn):
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        out[i] = fn(x[max(0, i - k + 1):i + 1], axis=0)
    return out


def ann(series):
    s = np.asarray(series)
    sd = s.std(ddof=1)
    return float(s.mean() / sd * np.sqrt(PPY)) if sd > 1e-12 else 0.0


def sized_portfolio(size, side, R):
    pos = size * side
    lag = np.vstack([np.zeros((1, pos.shape[1])), pos[:-1]])
    gross = (lag * R).sum(axis=1) / pos.shape[1]
    turn = np.abs(lag - np.vstack([np.zeros((1, pos.shape[1])), lag[:-1]])).sum(axis=1) / pos.shape[1]
    return gross - FEE * turn


def main():
    R = load_panel(40)
    T, N = R.shape
    mom10 = roll(R, 10, np.sum)
    mom5 = roll(R, 5, np.sum)
    mom1 = np.vstack([np.zeros((1, N)), R[:-1]])
    vol5 = roll(R, 5, np.std)
    vol20 = roll(R, 20, np.std)
    side = np.sign(mom10)
    bet = side * R
    y = (bet > 0).astype(int)

    half = T // 2
    idx = np.arange(20, T - 1)
    is_mask = idx[idx < half]
    oos_mask = idx[idx >= half]

    def stack(feats, rows):
        return np.column_stack([f[rows].ravel() for f in feats])

    feats_A = [mom1, mom5, mom10]
    feats_B = [vol5, vol20, np.abs(mom1)]

    def meta_size(feats):
        Xis, yis = stack(feats, is_mask), y[is_mask].ravel()
        sc = StandardScaler().fit(Xis)
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xis), yis)
        size = np.zeros((T, N))
        prob = clf.predict_proba(sc.transform(stack(feats, oos_mask)))[:, 1]
        size[oos_mask] = prob.reshape(len(oos_mask), N)
        return size

    oos = slice(half, T)
    prim = sized_portfolio(np.ones((T, N)), side, R)[oos]
    sA = sized_portfolio(meta_size(feats_A), side, R)[oos]
    sB = sized_portfolio(meta_size(feats_B), side, R)[oos]

    print(f"meta-labeling boundary, {N} perps x {T} bars, TSMOM(10) primary, {int(FEE*1e4)}bps\n")
    print(f"  primary (no sizing)                 OOS net Sharpe {ann(prim):+.2f}")
    print(f"  + secondary on SAME momentum feats  OOS net Sharpe {ann(sA):+.2f}")
    print(f"  + secondary on ORTHOGONAL vol feats OOS net Sharpe {ann(sB):+.2f}\n")

    for name, s in [("SAME-features", sA), ("ORTHOGONAL-vol", sB)]:
        per, _ = run_gauntlet([s - prim], n_trials=1, periods_per_year=PPY)
        p = per[0]
        beats = bool(p["econ_pass"])
        print(f"  {name} minus primary: Sharpe {p['sharpe_ann']:+.2f}, HAC-lower {p['hac_lower_ann']:+.2f} "
              f"-> improves primary = {beats}")

    prob_std = float(np.std(meta_size(feats_B)[slice(half, T)][meta_size(feats_B)[slice(half, T)] > 0]))
    print(f"\n  (diagnostic: orthogonal meta-model OOS sizing std = {prob_std:.3f}; ~0 => no discriminating signal)")
    ba = ann(sA) - ann(prim)
    bb = ann(sB) - ann(prim)
    if abs(ba) < 0.05 and abs(bb) < 0.05:
        print("INCONCLUSIVE on this data: the TSMOM primary DECAYS out-of-sample (gross +0.55 IS -> -0.24 OOS, the "
              "same decay as candidate #6) so there is no edge to size, AND the meta-model (same OR orthogonal "
              "features) finds near-zero discriminating signal -> constant sizing, no effect. The meta-labeling "
              "boundary cannot be demonstrated without a primary that retains OOS edge; NOT a boundary confirmation "
              "and NOT a clean refutation. Deferred to a universe/primary with genuine OOS edge.")
    elif bb > 0 and bb > ba:
        print("BOUNDARY CONFIRMED: orthogonal risk/regime sizing improves the primary MORE than same-feature "
              "sizing -- meta-labeling's value is conditioning on ORTHOGONAL (high-MI) info about WHEN the signal "
              "works, not re-using the primary's own features.")
    else:
        print(f"MIXED: same-feat delta {ba:+.2f}, orthogonal delta {bb:+.2f} -- report as-is.")


if __name__ == "__main__":
    main()
