"""METHOD bet, the decisive REAL-DATA cut. The stationary synthetic (complexity_mi_persistence.py) showed that in
a faithful-KMZ world complexity ALWAYS pays net-of-cost, rising with MI, with NO threshold and NO persistence
gate -- so the 'complexity fails' regime the KMZ-vs-Nagel debate is about must come from NON-STATIONARITY / weak
fleeting signal, which only real data has. Here we run the same faithful ladder (RFF + genuine ridge z on the
sample covariance, trainer.complexity_ladder) on real daily targets spanning the predictability axis:
  - ret1  : next-day return           (LOW MI, low persistence, tradeable)   -- the hard KMZ market-timing case
  - ret5  : next-5-day return         (low-mid MI, tradeable)
  - rvol5 : next-5-day realized vol    (HIGH MI, high persistence, measured by IC only, not a timing P&L)
Per asset: standardized lagged features (returns/momentum/realized-vol), strict TRAIN-only standardization, a
temporal 60/40 split (no leakage), the P-sweep ladder, OOS IC per (target,complexity) averaged across assets with
SE, plus a net-of-cost timing Sharpe for the return targets. Question: does the complexity premium (high-P minus
low-P OOS IC) SCALE with the target's real predictability, and does it SURVIVE cost for the tradeable targets?"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.complexity_ladder import kmz_ridge_fit_predict, random_fourier_features  # noqa: E402
from trainer.mutual_information import ksg_mi  # noqa: E402
from trainer.sharpe import sharpe_stats  # noqa: E402

UNI = "universe"
CLASSES = ("equity", "crypto", "commodity")
P_SWEEP = (4, 16, 64, 200, 600, 1500)
GAMMA = 0.5
Z = 1.0
ANN = math.sqrt(252.0)
COST_BPS = 5.0
MIN_DAYS = 800
SIMPLE_P = 4
COMPLEX_P = 600


def load_closes():
    out = []
    man = json.load(open(f"{UNI}/manifest.json"))
    for sym, meta in man.items():
        if meta["class"] not in CLASSES or meta["n"] < MIN_DAYS:
            continue
        d = json.load(open(f"{UNI}/{sym}.json"))["data"]
        dates = sorted(d)
        c = np.array([d[x][3] for x in dates], dtype=float)
        c = c[np.isfinite(c) & (c > 0)]
        if c.size >= MIN_DAYS:
            out.append((sym, meta["class"], c))
    return out


def features_targets(close):
    r = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])
    n = r.size
    feats, names = [], []

    def lag(a, k):
        return np.concatenate([[np.nan] * k, a[:-k]]) if k > 0 else a

    for k in (0, 1, 2, 4):
        feats.append(lag(r, k)); names.append(f"r{k}")
    for w in (5, 10, 20, 60):
        mom = np.full(n, np.nan)
        rv = np.full(n, np.nan)
        for t in range(w, n):
            seg = r[t - w + 1:t + 1]
            mom[t] = np.nanmean(seg)
            rv[t] = np.nanstd(seg)
        feats.append(mom); names.append(f"mom{w}")
        feats.append(rv); names.append(f"rv{w}")
    feats.append(np.abs(r)); names.append("absr")
    X = np.column_stack(feats)

    ret1 = np.concatenate([r[1:], [np.nan]])
    ret5 = np.full(n, np.nan)
    rvol5 = np.full(n, np.nan)
    for t in range(n - 5):
        seg = r[t + 1:t + 6]
        ret5[t] = np.nansum(seg)
        rvol5[t] = np.nanstd(seg)
    return X, {"ret1": ret1, "ret5": ret5, "rvol5": rvol5}


def ic(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30 or a[m].std() == 0 or b[m].std() == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def net_sharpe(pred, ret):
    m = np.isfinite(pred) & np.isfinite(ret)
    pred, ret = pred[m], ret[m]
    if pred.size < 50 or pred.std() == 0:
        return np.nan
    pos = np.clip(pred / pred.std(), -3, 3)
    turn = np.abs(np.diff(pos, prepend=pos[0]))
    net = pos * ret - COST_BPS * 1e-4 * turn
    return sharpe_stats(net)["sharpe"] * ANN


def run_asset(X, targets, rng):
    n = X.shape[0]
    split = int(n * 0.6)
    rows = np.arange(40, n - 6)
    tr = rows[rows < split]
    te = rows[rows >= split]
    if tr.size < 300 or te.size < 200:
        return None
    Xtr_raw, Xte_raw = X[tr], X[te]
    mu = np.nanmean(Xtr_raw, axis=0)
    sd = np.nanstd(Xtr_raw, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    Xtr = np.nan_to_num((Xtr_raw - mu) / sd)
    Xte = np.nan_to_num((Xte_raw - mu) / sd)
    d = X.shape[1]
    out = {tname: {} for tname in targets}
    sharpe = {tname: {} for tname in targets}
    for p in P_SWEEP:
        W = rng.standard_normal((d, p)) * math.sqrt(GAMMA)
        b = rng.uniform(0, 2 * math.pi, p)
        Ztr, Zte = random_fourier_features(Xtr, W, b), random_fourier_features(Xte, W, b)
        for tname, tvec in targets.items():
            ytr = tvec[tr]
            mtr = np.isfinite(ytr)
            if mtr.sum() < 200:
                out[tname][p] = np.nan
                continue
            yc = ytr[mtr] - ytr[mtr].mean()
            _, pred_te = kmz_ridge_fit_predict(Ztr[mtr], yc, Zte, Z)
            out[tname][p] = ic(pred_te, tvec[te])
            if tname in ("ret1", "ret5"):
                sharpe[tname][p] = net_sharpe(pred_te, tvec[te])
    return out, sharpe


def main():
    assets = load_closes()
    print(f"loaded {len(assets)} assets ({', '.join(sorted(set(c for _,c,_ in assets)))}); "
          f"faithful ladder z={Z}, cost={COST_BPS}bps, temporal 60/40\n")
    tnames = ("ret1", "ret5", "rvol5")
    ic_acc = {t: {p: [] for p in P_SWEEP} for t in tnames}
    sh_acc = {t: {p: [] for p in P_SWEEP} for t in ("ret1", "ret5")}
    persist = {t: [] for t in tnames}
    mi_proxy = {t: [] for t in tnames}
    for i, (sym, cls, close) in enumerate(assets):
        X, targets = features_targets(close)
        rng = np.random.default_rng(7000 + i)
        res = run_asset(X, targets, rng)
        if res is None:
            continue
        oos, sh = res
        for t in tnames:
            for p in P_SWEEP:
                v = oos[t].get(p, np.nan)
                if np.isfinite(v):
                    ic_acc[t][p].append(v)
            tv = targets[t]
            fin = tv[np.isfinite(tv)]
            if fin.size > 100:
                persist[t].append(float(np.corrcoef(fin[:-1], fin[1:])[0, 1]))
                mi_proxy[t].append(ksg_mi(X[:, 5][np.isfinite(tv) & np.isfinite(X[:, 5])],
                                          tv[np.isfinite(tv) & np.isfinite(X[:, 5])], k=5))
        for t in ("ret1", "ret5"):
            for p in P_SWEEP:
                v = sh[t].get(p, np.nan)
                if np.isfinite(v):
                    sh_acc[t][p].append(v)

    def stat(xs):
        xs = [x for x in xs if np.isfinite(x)]
        return (float(np.mean(xs)), float(np.std(xs, ddof=1) / math.sqrt(len(xs)))) if len(xs) > 2 else (np.nan, np.nan)

    print(f"=== OOS IC by complexity c=P/split (avg across assets +/- SE) ===")
    print(f"{'target':>7}{'persist':>9}{'MIproxy':>9}  " + "".join(f"{p:>9}" for p in P_SWEEP) + "   premium(hi-lo)")
    summary = {"n_assets": len(assets), "targets": {}}
    for t in tnames:
        row = {p: stat(ic_acc[t][p]) for p in P_SWEEP}
        prem_series = np.array(ic_acc[t][COMPLEX_P]) if ic_acc[t][COMPLEX_P] else np.array([])
        base_series = np.array(ic_acc[t][SIMPLE_P]) if ic_acc[t][SIMPLE_P] else np.array([])
        k = min(len(prem_series), len(base_series))
        prem = (prem_series[:k] - base_series[:k]) if k > 2 else np.array([np.nan])
        pm, pse = float(np.nanmean(prem)), float(np.nanstd(prem, ddof=1) / math.sqrt(max(k, 1)))
        star = "*" if (pm - 1.96 * pse) > 0 else (" (dead)" if (pm + 1.96 * pse) < 0 else "")
        per = float(np.nanmean(persist[t])); mip = float(np.nanmean(mi_proxy[t]))
        print(f"{t:>7}{per:>9.2f}{mip:>9.3f}  " + "".join(f"{row[p][0]:>9.3f}" for p in P_SWEEP)
              + f"   {pm:+.3f}{star}")
        summary["targets"][t] = {"persist": per, "mi_proxy": mip,
                                 "ic_by_p": {str(p): row[p][0] for p in P_SWEEP},
                                 "premium_hi_lo": pm, "premium_se": pse}

    print(f"\n=== NET-OF-COST timing Sharpe by complexity (return targets; {COST_BPS}bps) ===")
    print(f"{'target':>7}  " + "".join(f"{p:>9}" for p in P_SWEEP))
    for t in ("ret1", "ret5"):
        row = {p: stat(sh_acc[t][p]) for p in P_SWEEP}
        print(f"{t:>7}  " + "".join(f"{row[p][0]:>9.2f}" for p in P_SWEEP))
        summary["targets"][t]["net_sharpe_by_p"] = {str(p): row[p][0] for p in P_SWEEP}

    os.makedirs("experiments/results", exist_ok=True)
    json.dump(summary, open("experiments/results/complexity_real_targets.json", "w"), indent=1, default=str)
    print("\nwrote experiments/results/complexity_real_targets.json")


if __name__ == "__main__":
    main()
