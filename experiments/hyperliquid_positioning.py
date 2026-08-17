"""DATA-bet Phase 1 -- the CHEAP-first gate. Tests whether Hyperliquid positioning-PRESSURE extremes (premium =
mark-oracle, and funding) carry any NET-OF-COST predictive edge, before we pay for the S3 archive that holds the
distinctive layer (observed liquidations + wallet attribution). Thesis: crowded positioning mean-reverts -- fade
premium/funding extremes. If this cheap layer is a powered-null, the expensive S3 layer must clear a high bar to
be worth it; if it shows a pulse, S3 is justified.

Runs each signal through the SAME powered-null referee as the cross-asset battery (HAC Sharpe SE, one-sided
bound verdict, MDE, BH/BY FDR across the family) and charges honest taker cost. Includes the ex-ante lead
control demanded by the kill-criteria: a signal that only correlates with the CONTEMPORANEOUS bar (not the next
one) is a same-bar shadow, not an edge. Reads hyperliquid/<COIN>.json (see scripts/fetch_hyperliquid.py)."""
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from powered_battery import hac_verdict  # noqa: E402
from trainer.sharpe import benjamini_hochberg, benjamini_yekutieli, sharpe_stats  # noqa: E402

DATA = "hyperliquid"
BPS = 1e-4
TAKER_COST = 5.0 * BPS
LOOKBACK = 30
HORIZONS = (1, 3, 5)
COST_MULT = (0.0, 1.0, 2.0, 4.0)


def _daily_funding(funding):
    agg = {}
    for tms, rate, prem in funding:
        d = datetime.fromtimestamp(tms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        a = agg.setdefault(d, [0.0, 0.0, 0])
        a[0] += float(rate)
        a[1] += float(prem)
        a[2] += 1
    return {d: (v[0], v[1] / v[2]) for d, v in agg.items() if v[2] > 0}


def load_coin(path):
    rec = json.load(open(path))
    daily = rec["daily"]
    fund = _daily_funding(rec["funding"])
    dates = sorted(d for d in daily if d in fund)
    if len(dates) < LOOKBACK + 40:
        return None
    close = np.array([daily[d][3] for d in dates], dtype=float)
    ret = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])
    funding = np.array([fund[d][0] for d in dates], dtype=float)
    premium = np.array([fund[d][1] for d in dates], dtype=float)
    return {"coin": rec["coin"], "dates": dates, "ret": ret, "funding": funding, "premium": premium}


def zscore(x, lb):
    z = np.full(x.shape, np.nan)
    for t in range(lb, len(x)):
        w = x[t - lb:t]
        w = w[np.isfinite(w)]
        if w.size < lb // 2 or w.std() == 0:
            continue
        z[t] = (x[t] - w.mean()) / w.std()
    return z


def signal_position(z):
    return -np.clip(z, -3.0, 3.0) / 3.0


def smoothed_position(pos, h):
    eff = np.full(pos.shape, np.nan)
    for t in range(len(pos)):
        lo = max(0, t - h + 1)
        w = pos[lo:t + 1]
        w = w[np.isfinite(w)]
        if w.size:
            eff[t] = w.mean()
    return eff


def strat_returns(pos, ret, h, cost):
    eff = smoothed_position(pos, h)
    n = len(ret)
    out = np.full(n, np.nan)
    prev = 0.0
    for t in range(n - 1):
        p = eff[t]
        if not np.isfinite(p) or not np.isfinite(ret[t + 1]):
            continue
        turn = abs(p - prev)
        out[t + 1] = p * ret[t + 1] - cost * turn
        prev = p
    return out[np.isfinite(out)]


def ex_ante_lead(pos, ret):
    p = pos[:-1]
    fwd = ret[1:]
    cur = ret[:-1]
    m = np.isfinite(p) & np.isfinite(fwd) & np.isfinite(cur)
    p, fwd, cur = p[m], fwd[m], cur[m]
    if p.size < 30 or p.std() == 0:
        return {"n": int(p.size), "corr_ex_ante": float("nan"), "corr_contemp": float("nan")}
    ca = float(np.corrcoef(p, fwd)[0, 1]) if fwd.std() > 0 else float("nan")
    cc = float(np.corrcoef(p, cur)[0, 1]) if cur.std() > 0 else float("nan")
    return {"n": int(p.size), "corr_ex_ante": ca, "corr_contemp": cc}


def cross_sectional_factor(coins, h, cost):
    all_dates = sorted({d for c in coins for d in c["dates"]})
    idx = {c["coin"]: {d: i for i, d in enumerate(c["dates"])} for c in coins}
    pos_by_coin = {}
    for c in coins:
        z = zscore(c["premium"], LOOKBACK)
        pos_by_coin[c["coin"]] = smoothed_position(signal_position(z), h)
    rets = []
    prev_w = {}
    for di in range(len(all_dates) - 1):
        d, dn = all_dates[di], all_dates[di + 1]
        raw = {}
        for c in coins:
            if d in idx[c["coin"]] and dn in idx[c["coin"]]:
                ti = idx[c["coin"]][d]
                p = pos_by_coin[c["coin"]][ti]
                if np.isfinite(p):
                    raw[c["coin"]] = p
        if len(raw) < 3:
            prev_w = {}
            continue
        mean_p = np.mean(list(raw.values()))
        dem = {k: v - mean_p for k, v in raw.items()}
        gross = sum(abs(v) for v in dem.values())
        if gross == 0:
            prev_w = {}
            continue
        w = {k: v / gross for k, v in dem.items()}
        port = 0.0
        turn = 0.0
        for c in coins:
            k = c["coin"]
            wk = w.get(k, 0.0)
            if dn in idx[k] and (idx[k][dn]) < len(c["ret"]):
                rn = c["ret"][idx[k][dn]]
                if np.isfinite(rn):
                    port += wk * rn
            turn += abs(wk - prev_w.get(k, 0.0))
        rets.append(port - cost * turn)
        prev_w = w
    return np.array(rets, dtype=float)


def main():
    files = sorted(f for f in os.listdir(DATA) if f.endswith(".json") and f != "manifest.json")
    coins = [c for c in (load_coin(os.path.join(DATA, f)) for f in files) if c]
    print(f"loaded {len(coins)} coins: {', '.join(c['coin'] for c in coins)}")
    print(f"cost = {TAKER_COST/BPS:.1f}bps/side taker, lookback {LOOKBACK}d, econ Sharpe 0.5, "
          f"shared powered-null referee (HAC SE, 252-day annualization)\n")

    signals = {"premium": lambda c: c["premium"], "funding": lambda c: c["funding"]}
    cells = []
    print(f"{'coin':<7}{'sig':<9}{'h':>2}{'shp_ann':>9}{'lo':>7}{'hi':>7}{'mde':>6}{'p1':>7}  {'verdict':<13}"
          f"{'exAnte':>8}{'contemp':>9}")
    for c in coins:
        for sname, sfn in signals.items():
            z = zscore(sfn(c), LOOKBACK)
            pos = signal_position(z)
            lead = ex_ante_lead(pos, c["ret"])
            for h in HORIZONS:
                r = strat_returns(pos, c["ret"], h, TAKER_COST)
                if r.size < 60:
                    continue
                v = hac_verdict(r)
                cells.append({"coin": c["coin"], "signal": sname, "h": h, "kind": "ts",
                              "sharpe_ann": v["sharpe_ann"], "p_one": v["p_one"], "verdict": v["verdict"],
                              "lo_ann": v["lo_ann"], "hi_ann": v["hi_ann"], "mde_ann": v["mde_ann"],
                              "n": v["n_obs"], "corr_ex_ante": lead["corr_ex_ante"],
                              "corr_contemp": lead["corr_contemp"]})
                print(f"{c['coin']:<7}{sname:<9}{h:>2}{v['sharpe_ann']:>9.2f}{v['lo_ann']:>7.2f}{v['hi_ann']:>7.2f}"
                      f"{v['mde_ann']:>6.2f}{v['p_one']:>7.3f}  {v['verdict']:<13}"
                      f"{lead['corr_ex_ante']:>8.3f}{lead['corr_contemp']:>9.3f}")

    print(f"\n{'CROSS-SECTIONAL positioning factor (premium, market-neutral L/S, breadth-powered)':<70}")
    print(f"{'h':>2}{'shp_ann':>9}{'lo':>7}{'hi':>7}{'mde':>6}{'p1':>7}  {'verdict':<13}{'n':>6}")
    for h in HORIZONS:
        r = cross_sectional_factor(coins, h, TAKER_COST)
        if r.size < 60:
            continue
        v = hac_verdict(r)
        cells.append({"coin": "_XS_", "signal": "premium", "h": h, "kind": "xs",
                      "sharpe_ann": v["sharpe_ann"], "p_one": v["p_one"], "verdict": v["verdict"],
                      "lo_ann": v["lo_ann"], "hi_ann": v["hi_ann"], "mde_ann": v["mde_ann"], "n": v["n_obs"],
                      "corr_ex_ante": float("nan"), "corr_contemp": float("nan")})
        print(f"{h:>2}{v['sharpe_ann']:>9.2f}{v['lo_ann']:>7.2f}{v['hi_ann']:>7.2f}{v['mde_ann']:>6.2f}"
              f"{v['p_one']:>7.3f}  {v['verdict']:<13}{v['n_obs']:>6}")

    pvals = [c["p_one"] for c in cells]
    bh = benjamini_hochberg(pvals, q=0.05)
    by = benjamini_yekutieli(pvals, q=0.05)
    n_surv = sum(1 for c in cells if c["verdict"] == "survivor")
    n_pn = sum(1 for c in cells if c["verdict"] == "powered-null")
    n_inc = sum(1 for c in cells if c["verdict"] == "inconclusive")
    n_bh = sum(bh)
    n_by = sum(by)
    print(f"\n=== FAMILY ({len(cells)} cells) ===")
    print(f"per-cell verdict: {n_surv} survivor / {n_pn} powered-null / {n_inc} inconclusive")
    print(f"FDR across family: BH rejects {n_bh}, BY rejects {n_by} (q=0.05)")
    winners = [c for i, c in enumerate(cells) if bh[i]]
    if winners:
        print("BH-surviving cells:")
        for c in sorted(winners, key=lambda x: -x["sharpe_ann"]):
            print(f"  {c['coin']:<7}{c['signal']:<9}h={c['h']} shp={c['sharpe_ann']:.2f} p={c['p_one']:.4f} "
                  f"exAnte={c['corr_ex_ante']:.3f} contemp={c['corr_contemp']:.3f}")

    summary = {"n_coins": len(coins), "coins": [c["coin"] for c in coins], "n_cells": len(cells),
               "n_survivor": n_surv, "n_powered_null": n_pn, "n_inconclusive": n_inc,
               "n_bh_reject": n_bh, "n_by_reject": n_by, "cost_bps_side": TAKER_COST / BPS,
               "cells": cells}
    os.makedirs("experiments/results", exist_ok=True)
    json.dump(summary, open("experiments/results/hyperliquid_positioning.json", "w"), indent=1, default=str)
    print("\nwrote experiments/results/hyperliquid_positioning.json")


if __name__ == "__main__":
    main()
