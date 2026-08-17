"""The DECISIVE free-data test the Hyperliquid verification demanded: is there ANY free positioning edge once the
continuation direction is properly POWERED and orthogonalized to price trend? Runs on the broad Binance panel
(~500 USDT perps, premium-index = unclamped positioning pressure, ~6.5y). Reports three things the 12-coin HL cut
could not:
  (1) PANEL cross-sectional IC of premium vs next-day return (N = coins x days) -- the high-power test of whether
      premium carries ANY forward content, and in which direction (continuation>0 / reversion<0);
  (2) the tradeable cross-sectional factor Sharpe, BOTH signs, RAW and price-momentum-ORTHOGONALIZED, across
      horizons and a taker/maker cost sweep, through the shared powered-null referee (HAC SE, one-sided verdict,
      BH/BY FDR) -- with the HONEST MDE (time-series Sharpe MDE ~ (z+z)/sqrt(years) is set by YEARS, not breadth);
  (3) era splits, so a decayed vs durable edge is visible.
A powered-null here (upper bound < 0.5, IC~0) makes the free-data negative airtight and justifies paying for the
S3 archive; a survivor means a free positioning edge exists and S3 waits."""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from powered_battery import hac_verdict  # noqa: E402
from trainer.sharpe import benjamini_hochberg, benjamini_yekutieli  # noqa: E402

DATA = "cexperps"
PPY = 365.0
BPS = 1e-4
LOOKBACK = 30
MOM_LB = 30
HORIZONS = (1, 3, 5, 10)
COSTS = {"0bps": 0.0, "maker1.8": 1.8 * BPS, "taker4.5": 4.5 * BPS, "2x9": 9.0 * BPS}
MIN_DAYS = 250
MIN_XS = 20
ERAS = [("pre2021", "2019-01-01", "2021-01-01"), ("2021-2023", "2021-01-01", "2023-01-01"),
        ("2023-2026", "2023-01-01", "2027-01-01")]


def load_panel():
    files = sorted(f for f in os.listdir(DATA) if f.endswith(".json") and f != "manifest.json")
    prem, px = {}, {}
    for f in files:
        rec = json.load(open(os.path.join(DATA, f)))
        dates = sorted(set(rec["premium"]) & set(rec["price"]))
        if len(dates) < MIN_DAYS:
            continue
        sym = rec["sym"]
        prem[sym] = pd.Series({d: rec["premium"][d] for d in dates})
        px[sym] = pd.Series({d: rec["price"][d][3] for d in dates})
    PREM = pd.DataFrame(prem).sort_index()
    PX = pd.DataFrame(px).sort_index()
    PREM.index = pd.to_datetime(PREM.index)
    PX.index = pd.to_datetime(PX.index)
    return PREM, PX


def zscore_time(df, lb):
    m = df.rolling(lb, min_periods=lb // 2).mean()
    s = df.rolling(lb, min_periods=lb // 2).std()
    return (df - m) / s.replace(0.0, np.nan)


def xs_standardize(row):
    v = row.dropna()
    if v.size < MIN_XS or v.std() == 0:
        return pd.Series(np.nan, index=row.index)
    return (row - v.mean()) / v.std()


def build_weights(prem_z, mom_z, orthogonalize):
    W = pd.DataFrame(np.nan, index=prem_z.index, columns=prem_z.columns)
    for dt, s_row in prem_z.iterrows():
        s = s_row.dropna()
        if s.size < MIN_XS:
            continue
        if orthogonalize:
            m = mom_z.loc[dt].reindex(s.index)
            ok = s.index[m.notna()]
            if ok.size < MIN_XS:
                continue
            s, m = s.loc[ok], m.loc[ok]
            beta = np.cov(s, m)[0, 1] / m.var() if m.var() > 0 else 0.0
            s = s - beta * m
        d = s - s.mean()
        g = d.abs().sum()
        if g > 0:
            W.loc[dt, d.index] = d / g
    return W


def factor_series(W, RET, h, cost, sign=1.0):
    Wh = (W.rolling(h, min_periods=1).mean() if h > 1 else W) * sign
    held = Wh.shift(1)
    gross = (held * RET).sum(axis=1, min_count=1)
    turn = (held - Wh.shift(2)).abs().sum(axis=1, min_count=1)
    net = gross - cost * turn
    return net.dropna()


def panel_ic(sig, RET, lo=None, hi=None):
    fwd = RET.shift(-1)
    ics, days = [], []
    for dt, s_row in sig.iterrows():
        if (lo is not None and dt < pd.Timestamp(lo)) or (hi is not None and dt >= pd.Timestamp(hi)):
            continue
        s = s_row.dropna()
        f = fwd.loc[dt].reindex(s.index).dropna() if dt in fwd.index else pd.Series(dtype=float)
        common = s.index.intersection(f.index)
        if common.size < MIN_XS:
            continue
        a, b = s.loc[common], f.loc[common]
        if a.std() == 0 or b.std() == 0:
            continue
        ics.append(float(np.corrcoef(a, b)[0, 1]))
    ics = np.array(ics)
    if ics.size < 30:
        return {"mean_ic": float("nan"), "t": float("nan"), "n_days": int(ics.size), "ic_ir_ann": float("nan")}
    t = ics.mean() / (ics.std(ddof=1) / math.sqrt(ics.size))
    return {"mean_ic": float(ics.mean()), "t": float(t), "n_days": int(ics.size),
            "ic_ir_ann": float(ics.mean() / ics.std(ddof=1) * math.sqrt(PPY))}


def orthogonalize_signal(prem_z, mom_z):
    resid = prem_z.copy()
    for dt in prem_z.index:
        s = prem_z.loc[dt].dropna()
        m = mom_z.loc[dt].reindex(s.index)
        ok = s.index[m.notna()]
        if ok.size < MIN_XS:
            resid.loc[dt] = np.nan
            continue
        ss, mm = s.loc[ok], m.loc[ok]
        beta = np.cov(ss, mm)[0, 1] / mm.var() if mm.var() > 0 else 0.0
        resid.loc[dt] = np.nan
        resid.loc[dt, ok] = ss - beta * mm
    return resid


def main():
    PREM, PX = load_panel()
    RET = PX.pct_change()
    MOM = PX.pct_change(MOM_LB)
    prem_z = zscore_time(PREM, LOOKBACK)
    mom_z = MOM.apply(xs_standardize, axis=1)
    print(f"panel: {PREM.shape[1]} symbols x {PREM.shape[0]} days "
          f"({PREM.index[0].date()}..{PREM.index[-1].date()}), median coins/day="
          f"{int(prem_z.notna().sum(axis=1).median())}")
    mde_ann = None

    prem_orth = orthogonalize_signal(prem_z, mom_z)
    print("\n=== (1) PANEL cross-sectional IC (premium vs next-day return; N=coins x days; IC<0 => fade wins) ===")
    ic_full = {}
    for name, sig in [("premium_raw", prem_z), ("premium_orth_mom", prem_orth)]:
        ic = panel_ic(sig, RET)
        ic_full[name] = ic
        print(f"  {name:18s} mean_IC={ic['mean_ic']:+.5f}  t={ic['t']:+.2f}  "
              f"IC_IR_ann={ic['ic_ir_ann']:+.2f}  ({ic['n_days']} days)")
    print("  per-era (premium_orth_mom) -- decay check:")
    ic_eras = {}
    for ename, lo, hi in ERAS:
        ic = panel_ic(prem_orth, RET, lo, hi)
        ic_eras[ename] = ic
        print(f"    {ename:12s} mean_IC={ic['mean_ic']:+.5f}  t={ic['t']:+.2f}  ({ic['n_days']} days)")

    print("\n=== (2) tradeable cross-sectional factor, BOTH signs, net-of-cost (fade = short crowded/high premium) ===")
    print(f"{'sign':<13}{'variant':<10}{'h':>3}  " + "".join(f"{c:>11}" for c in COSTS)
          + f"  {'verdict@4.5':>13}{'MDEann':>8}")
    cells = []
    weights = {}
    for orth in (False, True):
        variant = "orth_mom" if orth else "raw"
        weights[variant] = build_weights(prem_z, mom_z, orth)
    for sign_name, sgn in (("fade", -1.0), ("continuation", 1.0)):
        for variant in ("raw", "orth_mom"):
            W = weights[variant]
            for h in HORIZONS:
                row_sh, vv = {}, None
                for cname, cost in COSTS.items():
                    net = factor_series(W, RET, h, cost, sign=sgn)
                    if net.size < 100:
                        row_sh[cname] = float("nan")
                        continue
                    v = hac_verdict(net, periods_per_year=PPY, sr_econ_ann=0.5)
                    row_sh[cname] = v["sharpe_ann"]
                    if cname == "taker4.5":
                        vv = v
                if vv is None:
                    continue
                mde_ann = vv["mde_ann"]
                cells.append({"sign": sign_name, "variant": variant, "h": h,
                              "sharpe_ann_taker": vv["sharpe_ann"], "lo": vv["lo_ann"], "hi": vv["hi_ann"],
                              "p_one": vv["p_one"], "verdict": vv["verdict"], "n": vv["n_obs"]})
                print(f"{sign_name:<13}{variant:<10}{h:>3}  " + "".join(f"{row_sh[c]:>11.2f}" for c in COSTS)
                      + f"  {vv['verdict']:>13}{vv['mde_ann']:>8.2f}")

    print("\n=== (3) era splits (FADE orth_mom h=5, taker 4.5bps) -- is the reversion edge decaying? ===")
    Wf = weights["orth_mom"]
    for ename, lo, hi in ERAS:
        net = factor_series(Wf, RET, 5, COSTS["taker4.5"], sign=-1.0)
        net = net[(net.index >= lo) & (net.index < hi)]
        if net.size < 60:
            print(f"  {ename:12s} n={net.size} too short")
            continue
        v = hac_verdict(net, periods_per_year=PPY, sr_econ_ann=0.5)
        print(f"  {ename:12s} Sharpe={v['sharpe_ann']:+.2f} CI[{v['lo_ann']:+.2f},{v['hi_ann']:+.2f}] "
              f"{v['verdict']:<13} n={v['n_obs']} MDEann={v['mde_ann']:.2f}")

    pvals = [c["p_one"] for c in cells]
    bh, by = benjamini_hochberg(pvals, 0.05), benjamini_yekutieli(pvals, 0.05)
    fade_surv = sum(1 for c in cells if c["sign"] == "fade" and c["verdict"] == "survivor")
    n_surv = sum(1 for c in cells if c["verdict"] == "survivor")
    n_pn = sum(1 for c in cells if c["verdict"] == "powered-null")
    print(f"\n=== FAMILY ({len(cells)} cells = 2 signs x 2 variants x {len(HORIZONS)} horizons) ===")
    print(f"per-cell: {n_surv} survivor ({fade_surv} fade) / {n_pn} powered-null / "
          f"{len(cells)-n_surv-n_pn} inconclusive")
    print(f"FDR: BH rejects {sum(bh)}, BY rejects {sum(by)} (q=0.05); "
          f"time-series MDE_ann~{mde_ann:.2f} (set by YEARS, not breadth)")
    summary = {"n_symbols": int(PREM.shape[1]), "n_days": int(PREM.shape[0]), "cells": cells,
               "n_survivor": n_surv, "n_survivor_fade": fade_surv, "n_powered_null": n_pn,
               "bh_reject": int(sum(bh)), "by_reject": int(sum(by)), "mde_ann": mde_ann,
               "panel_ic_full": ic_full, "panel_ic_eras": ic_eras}
    os.makedirs("experiments/results", exist_ok=True)
    json.dump(summary, open("experiments/results/positioning_factor_powered.json", "w"), indent=1, default=str)
    print("\nwrote experiments/results/positioning_factor_powered.json")


if __name__ == "__main__":
    main()
