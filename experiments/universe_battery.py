"""Powered-null battery on the WIDE + DEEP free universe (universe/ from scripts/fetch_universe.py) — the
power-maxed re-run. Same factor x asset-class cascade + HAC/within-cell-deflated powered verdict as
powered_battery.py, but on 50 equities x ~46y, ~18 commodities/16 bonds/8 fx x ~20y, 20 crypto x ~12y. The
bombshell test: with 20-46 years and full breadth, do the INCONCLUSIVE cells become POWERED nulls (an earned
"no 0.5-Sharpe edge") or does a survivor finally clear?"""
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_battery import FACTORS, weight_book, trend_book, factor_score  # noqa: E402
from powered_battery import hac_verdict, ANN  # noqa: E402
from trainer.sharpe import minimum_detectable_sharpe, benjamini_hochberg, benjamini_yekutieli, expected_max_sharpe  # noqa: E402

SP = os.path.dirname(os.path.abspath(__file__))
BS = os.path.dirname(SP)
UNI = os.path.join(BS, "universe")
RNG = np.random.default_rng(7)
CLASSES = ["equity", "crypto", "commodity", "bond", "fx"]
FEE = {"equity": 0.0005, "crypto": 0.0010, "commodity": 0.0005, "bond": 0.0003, "fx": 0.0003}


def load_universe(cls):
    """Per-class close panel (dates x symbols) + per-symbol (High, Low, dollarADV median)."""
    closes, hl = {}, {}
    for path in sorted(glob.glob(f"{UNI}/*.json")):
        if path.endswith("manifest.json"):
            continue
        rec = json.load(open(path))
        if rec.get("class") != cls:
            continue
        sym = os.path.basename(path)[:-5]
        d = rec["data"]
        idx = pd.DatetimeIndex([pd.Timestamp(k) for k in d])
        arr = np.array(list(d.values()), dtype=float)  # [o,h,l,c,v]
        closes[sym] = pd.Series(arr[:, 3], index=idx).sort_index()
        adv = np.nanmedian(arr[:, 3] * arr[:, 4]) if arr.shape[0] else np.nan
        hl[sym] = (pd.Series(arr[:, 1], index=idx).sort_index(), pd.Series(arr[:, 2], index=idx).sort_index(), adv)
    if not closes:
        return None, None
    PX = pd.DataFrame(closes).sort_index()
    PX = PX.dropna(how="all")
    return PX, hl


def run_class(cls):
    PX, hl = load_universe(cls)
    if PX is None or PX.shape[1] < 3:
        return []
    RET = PX.pct_change(fill_method=None)
    n = len(PX); split = int(n * 0.55); tr, te = PX.index[:split], PX.index[split:]
    k = 3 if PX.shape[1] >= 8 else 2 if PX.shape[1] >= 5 else 1
    fee = FEE[cls]
    rows = []
    for factor, lbs in FACTORS.items():
        lbs = [lb for lb in lbs if lb < split - 30]
        if not lbs:
            continue
        per = []
        for lb in lbs:
            if factor == "trend":
                W = trend_book(PX, lb, k)
            else:
                sc, inv = factor_score(PX, factor, lb)
                W = weight_book(sc, PX, inv, k)
            gross = (W.shift(1).fillna(0.0) * RET).sum(axis=1)
            net = gross - fee * W.diff().abs().sum(axis=1)
            tr_s = net.loc[tr].dropna(); te_s = net.loc[te].dropna()
            s_tr = tr_s.mean() / tr_s.std() if len(tr_s) > 30 and tr_s.std() > 0 else np.nan
            s_te = te_s.mean() / te_s.std() if len(te_s) > 30 and te_s.std() > 0 else np.nan
            per.append((lb, s_tr, s_te, net.loc[te].dropna()))
        per = [p for p in per if p[1] == p[1]]
        if not per:
            continue
        best = max(per, key=lambda p: p[1])
        trial_std = float(np.std([p[2] for p in per if p[2] == p[2]], ddof=1)) if len([p for p in per if p[2] == p[2]]) > 1 else 0.0
        v = hac_verdict(best[3], n_configs=len(per), trial_sr_std=trial_std)
        rows.append(dict(cls=cls, factor=factor, k=k, n_symbols=PX.shape[1], **v))
    return rows


def main():
    cells = []
    for cls in CLASSES:
        cells += run_class(cls)

    print("=" * 122)
    print("UNIVERSE POWERED BATTERY — wide+deep free panel; annualized deflated Sharpe, 90% one-sided CI, MDE, verdict @ SR_econ=0.5")
    print("=" * 122)
    print(f"{'class':10s}{'factor':12s}{'#sym':>5s}{'n':>7s}{'Sh_raw':>8s}{'Sh_adj':>8s}{'lo':>7s}{'hi':>7s}{'MDE':>6s}{'p1':>7s}  verdict")
    counts = {"survivor": 0, "powered-null": 0, "inconclusive": 0}
    for c in sorted(cells, key=lambda c: (c["cls"], -c["sr_adj_ann"])):
        counts[c["verdict"]] += 1
        print(f"{c['cls']:10s}{c['factor']:12s}{c['n_symbols']:>5d}{c['n_obs']:>7d}{c['sharpe_ann']:>8.2f}{c['sr_adj_ann']:>8.2f}"
              f"{c['lo_ann']:>7.2f}{c['hi_ann']:>7.2f}{c['mde_ann']:>6.2f}{c['p_one']:>7.3f}  {c['verdict']}")
    print(f"\n  verdict counts: {counts}   (cells={len(cells)})")

    pv = [c["p_one"] for c in cells]
    scopes = {"pooled BH": benjamini_hochberg(pv, 0.05), "pooled BY": benjamini_yekutieli(pv, 0.05)}
    rc = [False] * len(cells)
    for cls in set(c["cls"] for c in cells):
        idx = [i for i, c in enumerate(cells) if c["cls"] == cls]
        sub = benjamini_hochberg([cells[i]["p_one"] for i in idx], 0.05)
        for j, i in enumerate(idx):
            rc[i] = sub[j]
    scopes["per-class BH"] = rc
    print("  FDR survivors by scope:")
    for s, rej in scopes.items():
        print(f"    {s:14s}: {sum(rej)}  {[cells[i]['cls']+'/'+cells[i]['factor'] for i,r in enumerate(rej) if r]}")

    print("\n  per-class MDE (annualized Sharpe detectable @80% power) + calibrated power to flag a TRUE 0.5 Sharpe:")
    for cls in CLASSES:
        cc = [c for c in cells if c["cls"] == cls]
        if not cc:
            continue
        n = int(np.median([c["n_obs"] for c in cc]))
        hits = sum(1 for _ in range(300) if hac_verdict(pd.Series(RNG.standard_normal(n) + 0.5 / ANN))["verdict"] == "survivor")
        mde = minimum_detectable_sharpe(n) * ANN
        pnull = sum(1 for c in cc if c["verdict"] == "powered-null")
        print(f"    {cls:10s} n~{n:>6d} ({n/252:.0f}y) MDE {mde:.2f}  power@0.5 {hits/300:.0%}  powered-nulls {pnull}/{len(cc)}")

    json.dump({"cells": cells, "counts": counts, "fdr": {k: sum(v) for k, v in scopes.items()}},
              open(f"{SP}/universe_battery_results.json", "w"), indent=2, default=float)
    print("\nwrote universe_battery_results.json")


if __name__ == "__main__":
    main()
