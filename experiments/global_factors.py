"""GLOBAL cross-asset factor premia (Baltussen-style) at MAXIMUM power — the capstone of the power push.

For each canonical factor (momentum 252, value 1260, short-reversal 21, low-vol 60, trend 200) build the
long/short book WITHIN each asset class, vol-scale each class book to unit annualized vol, then average across
the 5 classes into ONE global factor over the union of dates. No lookback selection (canonical construction =
no within-cell multiplicity), so the whole sample is a valid evaluation — maximizing years and thus power. A
global factor diversified across 5 classes and 112 instruments over ~20-46y is the most powered test free data
can mount: if these are POWERED nulls, "no free-data factor premium clears a 0.5 net Sharpe" is earned, not
underpowered. Powered verdict via the HAC / deflated primitives; multiplicity across the ~5 factors via BH."""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_battery import weight_book, trend_book, factor_score  # noqa: E402
from powered_battery import hac_verdict, ANN  # noqa: E402
from universe_battery import load_universe, CLASSES, FEE  # noqa: E402
from trainer.sharpe import minimum_detectable_sharpe, benjamini_hochberg, benjamini_yekutieli  # noqa: E402

SP = os.path.dirname(os.path.abspath(__file__))
CANON = {"momentum": ("momentum", 252), "value": ("value", 1260), "st_reversal": ("st_reversal", 21),
         "lowvol": ("lowvol", 60), "trend": ("trend", 200)}
VOL_TARGET = 0.10 / math.sqrt(252.0)   # 10% annualized, in per-bar units


def class_book(PX, RET, fee, factor, lb, k):
    if factor == "trend":
        W = trend_book(PX, lb, k)
    else:
        sc, inv = factor_score(PX, factor, lb)
        W = weight_book(sc, PX, inv, k)
    net = (W.shift(1).fillna(0.0) * RET).sum(axis=1) - fee * W.diff().abs().sum(axis=1)
    return net


def vol_scale(series):
    s = series.dropna()
    sd = s.std()
    if not (sd > 0):
        return None
    return (series / sd) * VOL_TARGET


def main():
    # cache per-class panels
    panels = {}
    for cls in CLASSES:
        PX, _ = load_universe(cls)
        if PX is None or PX.shape[1] < 3:
            continue
        RET = PX.pct_change(fill_method=None)
        k = 3 if PX.shape[1] >= 8 else 2 if PX.shape[1] >= 5 else 1
        panels[cls] = (PX, RET, k)

    rows = []
    for gname, (factor, lb) in CANON.items():
        legs = {}
        for cls, (PX, RET, k) in panels.items():
            if lb >= len(PX) - 60:
                continue
            net = class_book(PX, RET, FEE[cls], factor, lb, k)
            vs = vol_scale(net)
            if vs is not None:
                legs[cls] = vs
        if len(legs) < 2:
            continue
        glob = pd.concat(legs.values(), axis=1).mean(axis=1)   # equal-risk average across active classes
        glob = glob.dropna()
        n_active = pd.concat(legs.values(), axis=1).notna().sum(axis=1)  # classes live per date
        v = hac_verdict(glob, n_configs=1)                     # canonical lookback -> no within-cell selection
        # era splits — is a survivor a pre-2004 equity-only artifact, or robust in the modern cross-asset era?
        eras = {}
        for lab, a, b in [("pre2010", "1900-01-01", "2009-12-31"), ("post2010", "2010-01-01", "2100-01-01"),
                          ("post2015", "2015-01-01", "2100-01-01")]:
            seg = glob[(glob.index >= a) & (glob.index <= b)]
            eras[lab] = hac_verdict(seg, n_configs=1) if len(seg) > 250 else {"verdict": "n/a", "sharpe_ann": float("nan")}
        rows.append(dict(factor=gname, n_classes=len(legs), classes=",".join(sorted(legs)),
                         multiclass_from=str(n_active[n_active >= 2].index.min())[:7], eras=eras, **v))

    print("=" * 108)
    print("GLOBAL CROSS-ASSET FACTOR PREMIA — vol-scaled, pooled across classes, FULL history (max power)")
    print("=" * 108)
    print(f"{'factor':12s}{'#cls':>5s}{'n':>7s}{'yrs':>5s}{'Sharpe':>8s}{'lo':>7s}{'hi':>7s}{'MDE':>6s}{'p1':>7s}  verdict   classes")
    for c in sorted(rows, key=lambda c: -c["sharpe_ann"]):
        print(f"{c['factor']:12s}{c['n_classes']:>5d}{c['n_obs']:>7d}{c['n_obs']/252:>5.0f}{c['sharpe_ann']:>8.2f}"
              f"{c['lo_ann']:>7.2f}{c['hi_ann']:>7.2f}{c['mde_ann']:>6.2f}{c['p_one']:>7.3f}  {c['verdict']:12s} {c['classes']}")
    print("\n  ERA SPLIT — Sharpe (verdict) by era; multiclass_from = first date >=2 classes live:")
    print(f"  {'factor':12s}{'≥2cls from':>12s}{'pre2010':>22s}{'post2010':>22s}{'post2015':>22s}")
    for c in sorted(rows, key=lambda c: -c["sharpe_ann"]):
        def cell(e):
            return f"{e['sharpe_ann']:+.2f} ({e['verdict'][:11]})" if e['sharpe_ann'] == e['sharpe_ann'] else "n/a"
        print(f"  {c['factor']:12s}{c['multiclass_from']:>12s}{cell(c['eras']['pre2010']):>22s}"
              f"{cell(c['eras']['post2010']):>22s}{cell(c['eras']['post2015']):>22s}")
    pv = [c["p_one"] for c in rows]
    bh = benjamini_hochberg(pv, 0.05); by = benjamini_yekutieli(pv, 0.05)
    print(f"\n  FDR over the {len(rows)} global factors: BH survivors {sum(bh)}  BY survivors {sum(by)}  "
          f"{[rows[i]['factor'] for i,r in enumerate(bh) if r]}")
    counts = {}
    for c in rows:
        counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
    print(f"  verdict counts: {counts}")
    print(f"  median MDE across global factors: {np.median([c['mde_ann'] for c in rows]):.2f} annualized Sharpe "
          f"(these are the most-powered free-data factor tests available).")
    json.dump({"global_factors": rows, "bh_survivors": sum(bh), "by_survivors": sum(by), "counts": counts},
              open(f"{SP}/global_factors_results.json", "w"), indent=2, default=float)
    print("\nwrote global_factors_results.json")


if __name__ == "__main__":
    main()
