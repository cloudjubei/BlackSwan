"""Robustness of the two global survivors (trend, momentum) BEFORE we believe them — cost sensitivity and
leave-one-class-out. A survivor that dies at realistic costs or rests on a single asset class is not a survivor."""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_battery import weight_book, trend_book, factor_score  # noqa: E402
from powered_battery import hac_verdict, ANN  # noqa: E402
from universe_battery import load_universe, CLASSES, FEE  # noqa: E402

CANON = {"momentum": ("momentum", 252), "trend": ("trend", 200)}
VOL_TARGET = 0.10 / math.sqrt(252.0)


def class_net(PX, RET, fee, factor, lb, k):
    if factor == "trend":
        W = trend_book(PX, lb, k)
    else:
        sc, inv = factor_score(PX, factor, lb)
        W = weight_book(sc, PX, inv, k)
    return (W.shift(1).fillna(0.0) * RET).sum(axis=1) - fee * W.diff().abs().sum(axis=1)


def vscale(s):
    sd = s.dropna().std()
    return (s / sd) * VOL_TARGET if sd > 0 else None


def build_global(panels, factor, lb, fee_mult=1.0, drop=None):
    legs = {}
    for cls, (PX, RET, k) in panels.items():
        if cls == drop or lb >= len(PX) - 60:
            continue
        vs = vscale(class_net(PX, RET, FEE[cls] * fee_mult, factor, lb, k))
        if vs is not None:
            legs[cls] = vs
    if len(legs) < 2:
        return None
    return pd.concat(legs.values(), axis=1).mean(axis=1).dropna()


def main():
    panels = {}
    for cls in CLASSES:
        PX, _ = load_universe(cls)
        if PX is None or PX.shape[1] < 3:
            continue
        RET = PX.pct_change(fill_method=None)
        k = 3 if PX.shape[1] >= 8 else 2 if PX.shape[1] >= 5 else 1
        panels[cls] = (PX, RET, k)

    for factor, (fac, lb) in CANON.items():
        print("=" * 84)
        print(f"GLOBAL {factor.upper()} robustness")
        print("=" * 84)
        print("  cost sensitivity (multiplier x base per-class fee):")
        for m in [1, 2, 4, 8]:
            g = build_global(panels, fac, lb, fee_mult=m)
            v = hac_verdict(g)
            print(f"    {m}x fee : Sharpe {v['sharpe_ann']:+.2f}  CI[{v['lo_ann']:+.2f},{v['hi_ann']:+.2f}]  -> {v['verdict']}")
        print("  leave-one-class-out (is it just one asset class?):")
        base = hac_verdict(build_global(panels, fac, lb))
        print(f"    all-5    : Sharpe {base['sharpe_ann']:+.2f}  -> {base['verdict']}")
        for cls in CLASSES:
            g = build_global(panels, fac, lb, drop=cls)
            if g is None:
                continue
            v = hac_verdict(g)
            print(f"    -{cls:9s}: Sharpe {v['sharpe_ann']:+.2f}  CI[{v['lo_ann']:+.2f},{v['hi_ann']:+.2f}]  -> {v['verdict']}")
        # post-2015, 2x cost — the hardest modern + realistic-cost test
        g = build_global(panels, fac, lb, fee_mult=2.0)
        seg = g[g.index >= "2015-01-01"]
        v = hac_verdict(seg)
        print(f"  HARDEST (post-2015, 2x cost): Sharpe {v['sharpe_ann']:+.2f}  CI[{v['lo_ann']:+.2f},{v['hi_ann']:+.2f}]  n={v['n_obs']}  -> {v['verdict']}")


if __name__ == "__main__":
    main()
